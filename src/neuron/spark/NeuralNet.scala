package neuron.spark

/**
  * Created by mao on 2016/4/16 0016.
  */

import breeze.linalg.{sum, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, svd => brzSvd}
import breeze.numerics.{sqrt, exp => Bexp, tanh => Btanh}
import breeze.stats.distributions.Rand
import neuron.util.{NNCheck, NNRunLog}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * label：目标矩阵
  * nna：神经网络每层节点的输出值,a(0),a(1),a(2)
  * error：输出层与目标值的误差矩阵
  */
case class NNLabel(label: Double, A: ArrayBuffer[BDV[Double]], error: Double) extends Serializable

/**
  * 配置参数
  */
case class NNConfig(size: Array[Int],
                    layer: Int,
                    activation_function: String,
                    learningRate: Double,
                    momentum: Double,
                    scaling_learningRate: Double,
                    weightPenaltyL2: Double,
                    nonSparsityPenalty: Double,
                    sparsityTarget: Double,
                    inputZeroMaskedFraction: Double,
                    dropoutFraction: Double,
                    testing: Int,
                    output_function: String) extends Serializable

/**
  * NN(neural network)
  */

class NeuralNet(private var size: Array[Int],
                private var layer: Int,
                private var activation_function: String,
                private var learningRate: Double,
                private var momentum: Double,
                private var scaling_learningRate: Double,
                private var weightPenaltyL2: Double,
                private var nonSparsityPenalty: Double,
                private var sparsityTarget: Double,
                private var inputZeroMaskedFraction: Double,
                private var dropoutFraction: Double,
                private var testing: Int,
                private var output_function: String,
                private var initW: Array[BDM[Double]]) extends Serializable with Logging {

  def this() = this(NeuralNet.Architecture, 3, NeuralNet.Activation_Function, 2.0, 0.5, 1.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0, NeuralNet.Output, Array(BDM.zeros[Double](1, 1)))

  /** 设置神经网络结构. Default: [10, 5, 1]. */
  def setSize(size: Array[Int]): this.type = {
    this.size = size
    this
  }

  /** 设置神经网络层数据. Default: 3. */
  def setLayer(layer: Int): this.type = {
    this.layer = layer
    this
  }

  /** 设置隐含层函数. Default: sigm. */
  def setActivation_function(activation_function: String): this.type = {
    this.activation_function = activation_function
    this
  }

  /** 设置学习率因子. Default: 2. */
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  /** 设置Momentum. Default: 0.5. */
  def setMomentum(momentum: Double): this.type = {
    this.momentum = momentum
    this
  }

  /** 设置scaling_learningRate. Default: 1. */
  def setScaling_learningRate(scaling_learningRate: Double): this.type = {
    this.scaling_learningRate = scaling_learningRate
    this
  }

  /** 设置正则化L2因子. Default: 0. */
  def setWeightPenaltyL2(weightPenaltyL2: Double): this.type = {
    this.weightPenaltyL2 = weightPenaltyL2
    this
  }

  /** 设置权重稀疏度惩罚因子. Default: 0. */
  def setNonSparsityPenalty(nonSparsityPenalty: Double): this.type = {
    this.nonSparsityPenalty = nonSparsityPenalty
    this
  }

  /** 设置权重稀疏度目标值. Default: 0.05. */
  def setSparsityTarget(sparsityTarget: Double): this.type = {
    this.sparsityTarget = sparsityTarget
    this
  }

  /** 设置权重加入噪声因子. Default: 0. */
  def setInputZeroMaskedFraction(inputZeroMaskedFraction: Double): this.type = {
    this.inputZeroMaskedFraction = inputZeroMaskedFraction
    this
  }

  /** 设置权重Dropout因子. Default: 0. */
  def setDropoutFraction(dropoutFraction: Double): this.type = {
    this.dropoutFraction = dropoutFraction
    this
  }

  /** 设置testing. Default: 0. */
  def setTesting(testing: Int): this.type = {
    this.testing = testing
    this
  }

  /** 设置输出函数. Default: linear. */
  def setOutput_function(output_function: String): this.type = {
    this.output_function = output_function
    this
  }

  /** 设置初始权重. Default: 0. */
  def setInitW(initW: Array[BDM[Double]]): this.type = {
    this.initW = initW
    this
  }

  /**
    * 运行神经网络算法.
    */
  def NNTrain(sampleData: RDD[(Double, BDV[Double])], params: Array[Double]): NeuralNetModel = {
    val sc = sampleData.sparkContext
    var startTime = System.currentTimeMillis()
    var endTime = System.currentTimeMillis()

    // 样本数据划分：训练数据、交叉检验数据
    val validation = params(1)
    val splitWeights = Array(1.0 - validation, validation)
    val sampleDataSplitArray = sampleData.randomSplit(splitWeights)
    val dataForTrain = sampleDataSplitArray(0)
    val dataForValid = sampleDataSplitArray(1)

    // sampleSize:训练样本的总量
    val sampleSize = dataForTrain.count
    //训练次数
    val epochs = params(0).toInt

    println("样本总量：" + sampleSize + ";训练次数：" + epochs)

    // 训练参数配置
    var nnConfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
      output_function)
    //交叉验证参数配置
    val evalConfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1,
      output_function)
    //参数配置广播
    var broadcastConfig = sc.broadcast(nnConfig)
    val broadcastEvalConfig = sc.broadcast(evalConfig)
    //权重配置
    var nn_W = NeuralNet.InitialWeight(size)
    var nn_vW = NeuralNet.InitialWeightV(size)
    //权重广播
    var broadcastW = sc.broadcast(nn_W)
    var broadcastVW = sc.broadcast(nn_vW)

    var minLoss = 10000.0
    var lossTrain = 0.0
    var lossValid = 0.0

    var minW = nn_W
    // 进行epochs次训练
    var isLocalMin = false
    var isConvergence = false
    var lastLoss = 1000.0
    var smallTime = 0
    var i = 0
    val log = ArrayBuffer[(Int, Long, Double, Double)]()
    while (i < epochs && !isConvergence) {
      //达到局部最小值后更改权重
      if (isLocalMin) {
        nn_W = NeuralNet.InitialWeight(size)
        nn_vW = NeuralNet.InitialWeightV(size)
        broadcastW = sc.broadcast(nn_W)
        broadcastVW = sc.broadcast(nn_vW)
        nnConfig = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
          weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, testing,
          output_function)
        //参数配置广播
        broadcastConfig = sc.broadcast(nnConfig)
        isLocalMin = false
        smallTime = 0
      }
      //第一次迭代前计算误差
      if (i == 0) {
        startTime = System.currentTimeMillis()
        lossTrain = NeuralNet.NNeval(dataForTrain, broadcastConfig, broadcastW)
        if (validation > 0) lossValid = NeuralNet.NNeval(dataForValid, broadcastConfig, broadcastW)
        // 训练前误差
        endTime = System.currentTimeMillis()
        printf("迭代前误差  Took = %d ms; Full-batch train mse = %f, val mse = %f.\n",
          endTime - startTime, lossTrain, lossValid)
        lastLoss = lossValid
      }

      // 在迭代训练样本中加入噪声
      val batch = if (broadcastConfig.value.inputZeroMaskedFraction != 0) {
        NeuralNet.AddNoise(dataForTrain, broadcastConfig.value.inputZeroMaskedFraction)
      } else dataForTrain
      // 前向传播
      val ffResult = NeuralNet.NNff(batch, broadcastConfig, broadcastW)
      // 后向传播
      val bpResult = NeuralNet.NNbp(ffResult, broadcastConfig, broadcastW)

      // 更新权重参数,并将其广播到各个节点
      val gradientResult = NeuralNet.NNapplygrads(bpResult, broadcastConfig, broadcastW, broadcastVW)
      nn_W = gradientResult(0)
      nn_vW = gradientResult(1)
      broadcastW = sc.broadcast(nn_W)
      broadcastVW = sc.broadcast(nn_vW)

      // 计算本次迭代的训练误差及交叉检验误差
      lossTrain = NeuralNet.NNeval(dataForTrain, broadcastEvalConfig, broadcastW)
      if (validation > 0) lossValid = NeuralNet.NNeval(dataForValid, broadcastEvalConfig, broadcastW)

      // 更新学习因子,并将其广播到各个节点用于下次迭代
      nnConfig = NNConfig(size, layer, activation_function, nnConfig.learningRate * nnConfig.scaling_learningRate,
        momentum, scaling_learningRate, weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction,
        dropoutFraction, testing,
        output_function)
      broadcastConfig = sc.broadcast(nnConfig)

      endTime = System.currentTimeMillis()
      // 打印输出结果
      printf("迭代次数= %d , Took = %d ms; Full-batch train mse = %f, val mse = %f.\n",
        i+1, endTime - startTime, lossTrain, lossValid)

      val r = (i+1, endTime - startTime, lossTrain, lossValid)
      log += r

      //记录最佳结果
      if (lossValid < minLoss) {
        minW = nn_W
        minLoss = lossValid
        if (lossValid < 0.0012) {
          isConvergence = true
          println("Convergence!! lossValid is %f", lossValid)
        }
      }
      if (lastLoss - lossValid >= -0.000000001 && lastLoss - lossValid <= 0.000000001) {
        smallTime += 1
        if (smallTime > 10) {
          isLocalMin = true
          println("local min!!")
        }
      } else {
        smallTime = 0
      }

      lastLoss = lossValid
      i += 1
    }
    NNRunLog.logProcess(log.toArray)
    val configOkay = NNConfig(size, layer, activation_function, learningRate, momentum, scaling_learningRate,
      weightPenaltyL2, nonSparsityPenalty, sparsityTarget, inputZeroMaskedFraction, dropoutFraction, 1,
      output_function)
    new NeuralNetModel(configOkay, minW)
  }

}

/**
  * NN(neural network)
  */
object NeuralNet extends Serializable {
  val Activation_Function = "sigm"
  val Output = "sigm"
  val Architecture = Array(10, 5, 1)

  /**
    * 增加随机噪声，把训练样例中的一些数据调整变为0
    * 若随机值>=fraction，值不变，否则改为0
    * 参见《Extracting and Composing Robust Features with DeNoising AutoEncoders》
    * @param fraction: 调整的比例
    */
  def AddNoise(rdd: RDD[(Double, BDV[Double])], fraction: Double): RDD[(Double, BDV[Double])] = {
    val addNoise = rdd.map { f =>
      val features = f._2
      val a = BDV.rand[Double](features.length)
      val a1 = a :>= fraction
      val d1 = a1.map { f => if (f) 1.0 else 0.0 }
      val features2 = features :* d1
      (f._1, features2)
    }
    addNoise
  }

  /**
    * 初始化权重
    * 初始化为一个很小的、接近零的随机值
    */
  def InitialWeight(size: Array[Int]): Array[BDM[Double]] = {
    val weightLists = ArrayBuffer[BDM[Double]]()
    for (i <- 1 until size.length) {
      val w = BDM.rand(size(i), size(i - 1) + 1, new Rand[Double] {
        def draw = Random.nextDouble()/10
      })
      w :-= 0.05
      weightLists += w
    }
    weightLists.toArray
  }

  /**
    * 初始化权重vW
    * 初始化为0
    */
  def InitialWeightV(size: Array[Int]): Array[BDM[Double]] = {
    val nn_vW = ArrayBuffer[BDM[Double]]()
    for (i <- 1 until size.length) {
      val d1 = BDM.zeros[Double](size(i), size(i - 1) + 1)
      nn_vW += d1
    }
    nn_vW.toArray
  }

  /**
    * 初始每一层的平均激活度
    * 初始化为0
    */
  def InitialActiveP(size: Array[Int]): Array[BDM[Double]] = {
    // 初始每一层的平均激活度
    // average activations (for use with sparsity)
    // nn.p{i}     = zeros(1, nn.size(i));
    val n = size.length
    val nn_p = ArrayBuffer[BDM[Double]]()
    nn_p += BDM.zeros[Double](1, 1)
    for (i <- 1 until n) {
      val d1 = BDM.zeros[Double](1, size(i))
      nn_p += d1
    }
    nn_p.toArray
  }

  /**
    * 随机让网络某些隐含层节点的权重不工作
    * 若随机值>=Fraction，矩阵值不变，否则改为0
    */
  def   DropoutWeight(vector: BDV[Double], Fraction: Double): Array[BDV[Double]] = {
    val aa = BDV.rand[Double](vector.length)
    val aa1 = aa :> Fraction
    val d1 = aa1.map { f => if (f) 1.0 else 0.0 }
    val vector2 = vector :* d1
    Array(d1, vector2)
  }

  /**
    * sigm激活函数
    * X = 1./(1+exp(-P));
    */
  def sigm(vector: BDV[Double]): BDV[Double] = {
    val s1 = 1.0 / (Bexp(vector * (-1.0)) + 1.0)
    s1
  }

  /**
    * tanh激活函数
    * f=1.7159*tanh(2/3.*A);
    */
  def tanh_opt(vector: BDV[Double]): BDV[Double] = {
    val s1 = Btanh(vector * (2.0 / 3.0)) * 1.7159
    s1
  }

  /**
    * nnff是进行前向传播
    * 计算神经网络中的每个节点的输出值;
    */
  def NNff(
            batch_xy2: RDD[(Double, BDV[Double])],
            bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
            bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): RDD[(NNLabel, Array[BDV[Double]])] = {
    // 第1层:a(1)=[1 x]
    // 增加偏置项b
    val ff1 = batch_xy2.map { f =>
      val label = f._1
      val x = f._2
      val A = ArrayBuffer[BDV[Double]]()
      val xb = BDV.vertcat(BDV.ones[Double](1), x)
      val error = 0.0
      A += xb
      NNLabel(label, A, error)
    }

    // feed forward pass
    // 第2至l-1层计算，a(i)=f(a(i-1)*w(i-1)')
    val ffj = ff1.map { f =>
      val A = f.A
      val dropOutMask = ArrayBuffer[BDV[Double]]()
      dropOutMask += new BDV[Double](Array(0.0))
      for (j <- 1 until bc_config.value.layer - 1) {
        // 计算每层输出
        // nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}')
        // nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');

        //Ai为前一层的输出，是一个向量
        val Ai = A(j - 1)
        //Wi为当前层与前一层之间的权重矩阵
        val Wi = bc_nn_W.value(j - 1)
        //Zj是当前层的
        val Zj = Wi * Ai
        val Aj = bc_config.value.activation_function match {
          case "sigm" =>
            NeuralNet.sigm(Zj)
          case "tanh_opt" =>
            NeuralNet.tanh_opt(Zj)
        }
        // dropout计算
        // Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分
        // 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
        // 参照 http://www.cnblogs.com/tornadomeet/p/3258122.html
        val dropoutAj = if (bc_config.value.dropoutFraction > 0) {
          if (bc_config.value.testing == 1) {
            val Aj2 = Aj * (1.0 - bc_config.value.dropoutFraction)
            Array(new BDV[Double](Array(0.0)), Aj2)
          } else {
            NeuralNet.DropoutWeight(Aj, bc_config.value.dropoutFraction)
          }
        } else {
          Array(new BDV[Double](Array(0.0)), Aj)
        }
        val Aj2 = dropoutAj(1)
        dropOutMask += dropoutAj(0)
        // 增加偏置项b
        val Aj3 = BDV.vertcat(BDV.ones[Double](1), Aj2)
        A += Aj3
      }
      (NNLabel(f.label, A, f.error), dropOutMask.toArray)
    }

    // 输出层计算
    val ffl = ffj.map { f =>
      val A = f._1.A
      // nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
      // nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
      val Al_ = A(bc_config.value.layer - 2)
      val Wl_ = bc_nn_W.value(bc_config.value.layer - 2)
      val Zl:BDV[Double] =  Wl_ * Al_
      val Al = bc_config.value.output_function match {
        case "sigm" =>
          NeuralNet.sigm(Zl)
        case "linear" =>
          Zl
      }
      A += Al
      (NNLabel(f._1.label, A, f._1.error), f._2)
    }

    // error and loss
    // 输出误差计算
    // nn.e = y - nn.a{n};
    // val nn_e = batch_y - Al
    val ffResult = ffl.map { f =>
      val batch_y = f._1.label
      val Al = f._1.A(bc_config.value.layer - 1)
      val error = batch_y - Al(0)
      (NNLabel(batch_y, f._1.A, error), f._2)
    }
    val nnlabel = ffResult.map(f => f._1)

    ffResult
  }




  /**
    * sparsity计算，网络稀疏度
    * 计算每个隐藏节点的平均活跃度
    */
  def activeP(ffResult: RDD[(NNLabel, Array[BDV[Double]])],
              bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
              nn_p_old: Array[BDV[Double]]): Array[BDV[Double]] = {
    val nn_p = ArrayBuffer[BDV[Double]]()
    nn_p += BDV.zeros[Double](1)
    // calculate running exponential activations for use with sparsity
    // sparsity计算，计算sparsity，nonSparsityPenalty 是对没达到sparsity target的参数的惩罚系数
    for (i <- 1 until bc_config.value.layer - 1) {
      val pi = ffResult.map(f => f._1.A(i))
      val zeroPi = BDV.zeros[Double](bc_config.value.size(i))
      val (piSum, batchSize) = pi.treeAggregate((zeroPi, 0L))(
        seqOp = (c, v) => {
          // c: (nnaSum, count), v: (nna)
          val nna1 = c._1
          val nna2 = v
          val nnaSum = nna1 + nna2
          (nnaSum, c._2 + 1)
        },
        combOp = (c1, c2) => {
          // c: (nnaSum, count)
          (c1._1 + c2._1, c1._2 + c2._2)
        })
      val piAvg = piSum / batchSize.toDouble
      val oldPi = nn_p_old(i)
      val newPi = (piAvg * 0.01) + (oldPi * 0.99)
      nn_p += newPi
    }
    nn_p.toArray
  }



  /**
    * NNbp是后向传播
    * 计算权重的平均偏导数
    */
  def NNbp(
            ffResult: RDD[(NNLabel, Array[BDV[Double]])],
            bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
            bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[BDM[Double]] = {
    // 第n层偏导数：d(n)=-(y-a(n))*f'(z)，sigmoid导函数表达式:f'(z)=f(z)*[1-f(z)]
    // {'softmax','linear'}: d{n} = - nn.e;
    val Dn = ffResult.map { f =>
      val A = f._1.A
      val error = f._1.error
      val D = ArrayBuffer[BDV[Double]]()
      val dn = bc_config.value.output_function match {
        case "sigm" =>
          val fz = A(bc_config.value.layer - 1)
          (error * (-1.0)) * (fz :* (1.0 - fz))
        case "linear" =>
          new BDV(Array(error * (-1.0)))
      }
      D += dn
      (f._1, f._2, D)
    }
    // 第n-1至第2层导数：d(n)=-(w(n)*d(n+1))*f'(z)
    val Di = Dn.map { f =>
      // 假设 f(z) 是sigmoid函数 f(z)=1/[1+e^(-z)]，f'(z)表达式，f'(z)=f(z)*[1-f(z)]
      // 假设 f(z) tanh f(z)=1.7159*tanh(2/3.*A) ，f'(z)表达式，f'(z)=1.7159 * 2/3 * (1 - 1/(1.7159)^2 * f(z).^2)
      val A = f._1.A
      val D = f._3
      val dropout = f._2
      for (i <- (bc_config.value.layer - 2) to 1 by -1) {
        // f'(z)表达式
        val nnd_act = bc_config.value.activation_function match {
          case "sigm" =>
            A(i) :* (1.0 - A(i))
          case "tanh_opt" =>
            val fz2 = 1.0 - ((A(i) :* A(i)) * (1.0 / (1.7159 * 1.7159)))
            fz2 * (1.7159 * (2.0 / 3.0))
        }

        // 导数：d(n)=-( w(n)*d(n+1)+ sparsityError )*f'(z)
        // d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act;
        val Wi = bc_nn_W.value(i)
        val nndi1 = if (i + 1 == bc_config.value.layer - 1) {
          //in this case in d{n} there is not the bias term to be removed
          val di1 = D(bc_config.value.layer - 2 - i)
          val di2 =  Wi.t * di1
          val di3 = di2 :* nnd_act
          di3
        } else {
          // in this case in d{i} the bias term has to be removed
          val di1 = D(bc_config.value.layer - 2 - i)(1 to -1)
          val di2 = Wi.t * di1
          val di3 = di2 :* nnd_act
          di3
        }
        // dropoutFraction
        val nndi2 = if (bc_config.value.dropoutFraction > 0) {
          val dropouti1 = dropout(i)
          val dropouti2 = BDV.vertcat(BDV.ones[Double](1), dropouti1)
          nndi1 :* dropouti2
        } else nndi1
        D += nndi2
      }
      D += BDV.zeros(1)
      // 计算最终需要的偏导数值
      // dW{i} = d{i + 1} * a{i}
      val dW = ArrayBuffer[BDM[Double]]()
      for (i <- 0 to bc_config.value.layer - 2) {
        val dwi = if (i + 1 == bc_config.value.layer - 1) {
          D(bc_config.value.layer - 2 - i) * A(i).t
        } else {
          D(bc_config.value.layer - 2 - i)(1 to -1) * A(i).t
        }
        dW += dwi
      }
      (f._1, D, dW.toArray)
    }

    val bpResult = Di.map(f => (f._1, f._3))
    //NNCheck.checkGradient(bpResult, bc_config.value, bc_nn_W.value)

    val dW = Di.map(f => f._3)
    // Sample a subset (fraction miniBatchFraction) of the total data
    // compute and sum up the subgradients on this subset (this is one map-reduce)
    val initGradList = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to bc_config.value.layer - 2) {
      initGradList += BDM.zeros[Double](bc_config.value.size(i + 1), bc_config.value.size(i) + 1)
    }
    val (gradientSum, miniBatchSize) = dW.treeAggregate((initGradList, 0L))(
      seqOp = (c, v) => {
        // c: (grad, count), v: (grad)
        val grad1 = c._1
        val grad2 = v
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (grad, count)
        val grad1 = c1._1
        val grad2 = c2._1
        val sumgrad = ArrayBuffer[BDM[Double]]()
        for (i <- 0 to bc_config.value.layer - 2) {
          val Bm1 = grad1(i)
          val Bm2 = grad2(i)
          val Bmsum = Bm1 + Bm2
          sumgrad += Bmsum
        }
        (sumgrad, c1._2 + c2._2)
      })
    NNRunLog.logWeight(gradientSum.toArray, "权重导数" + miniBatchSize)
    // 求平均值
    val gradientAvg = ArrayBuffer[BDM[Double]]()
    for (i <- 0 until bc_config.value.layer - 1) {
      val Bm1 = gradientSum(i)
      val Bmavg = Bm1 :/ miniBatchSize.toDouble
      gradientAvg += Bmavg
    }
    gradientAvg.toArray
  }

  /**
    * NNapplygrads是权重更新
    * 权重更新
    */
  def NNapplygrads(
                    avgGradList: Array[BDM[Double]],
                    bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
                    bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]],
                    bc_nn_vW: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Array[Array[BDM[Double]]] = {
    // nn = nnapplygrads(nn) returns an neural network structure with updated
    // weights and biases
    // 更新权重参数：w=w-α*[dw + λw]
    val W_a = ArrayBuffer[BDM[Double]]()
    val vW_a = ArrayBuffer[BDM[Double]]()
    for (i <- 0 until bc_config.value.layer - 1) {
      val nndwi = if (bc_config.value.weightPenaltyL2 > 0) {
        val dwi = avgGradList(i)
        val Wi = bc_nn_W.value(i)
        val zeros = BDM.zeros[Double](dwi.rows, 1)
        val l2 = BDM.horzcat(zeros, Wi(::, 1 to -1))
        val dwi2 = dwi + (l2 * bc_config.value.weightPenaltyL2)
        dwi2
      } else {
        avgGradList(i)
      }
      val nndwi2 = nndwi * bc_config.value.learningRate
      val nndwi3 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val dw3 = nndwi2 + (vwi * bc_config.value.momentum)
        dw3
      } else {
        nndwi2
      }
      // nn.W{i} = nn.W{i} - dW;
      W_a += (bc_nn_W.value(i) - nndwi3)
      // nn.vW{i} = nn.momentum*nn.vW{i} + dW;
      val nnvwi1 = if (bc_config.value.momentum > 0) {
        val vwi = bc_nn_vW.value(i)
        val vw3 = nndwi2 + (vwi * bc_config.value.momentum)
        vw3
      } else {
        bc_nn_vW.value(i)
      }
      vW_a += nnvwi1
    }
    Array(W_a.toArray, vW_a.toArray)
  }

  /**
    * 进行前向传播计算代价函数值
    * 计算神经网络中的每个节点的输出值，并计算平均误差;
    */
  def NNeval(
              batch_xy: RDD[(Double, BDV[Double])],
              bc_config: org.apache.spark.broadcast.Broadcast[NNConfig],
              bc_nn_W: org.apache.spark.broadcast.Broadcast[Array[BDM[Double]]]): Double = {
    // NNff是进行前向传播
    // nn = nnff(nn, batch_x, batch_y);
    val ffResult = NeuralNet.NNff(batch_xy, bc_config, bc_nn_W)
    NNRunLog.logAn(ffResult.map(f => f._1), "各层输出")
    // error and loss
    // 输出误差计算
    val loss1 = ffResult.map(f => f._1.error)
   loss1.map(d => d*d).mean() * 0.5
  }
}
