package neuron.test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{*, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, max => Bmax, min => Bmin, sum => Bsum, svd => brzSvd}
import neuron.spark.NeuralNet
import neuron.util.{NNRunLog, RandomSampleData}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by mao on 2016/4/16 0016.
  */
object TestForNN {
  def main(args: Array[String]) {

    /*构建Spark对象*/
    val conf = new SparkConf().setAppName("NNtest").setMaster("spark://spark4:7077").setJars(List("C:\\Users\\cblk\\IdeaProjects\\NeuronNetworks\\out\\artifacts\\NeuronNetworks_jar\\NeuronNetworks.jar"))
    val sc = new SparkContext(conf)
    //val nameNode = "hdfs://10.141.211.123:9000/"

    /*随机生成测试数据*/
    // 生成原始随机样本数据
    Logger.getRootLogger.setLevel(Level.WARN)
    val sampleNum = 100
    val xDimens = 5
    val sampleMatrix = RandomSampleData.RandM(sampleNum, xDimens, -10, 10, "sphere")
    // 归一化
    val normMax = Bmax(sampleMatrix(::, *))
    val normMin = Bmin(sampleMatrix(::, *))
    val norm1 = sampleMatrix - BDM.ones[Double](sampleMatrix.rows, 1) * normMin
    val norm2 = norm1 :/ (BDM.ones[Double](norm1.rows, 1) * (normMax - normMin))
    // 将矩阵形式的原始样本拆分成数组形式保存
    val sampleArray = ArrayBuffer[BDM[Double]]()
    for (i <- 0 until sampleNum) {
      val mi = norm2(i, ::).inner.toArray
      val y_x = new BDM(1, mi.length, mi)
      sampleArray += y_x
    }
    //由本地的样本数据集合生成分布式内存数据集Rdd
    val sampleRdd = sc.parallelize(sampleArray, 10)
    //sc.setCheckpointDir(nameNode + "checkpoint")
    //sampleRdd.checkpoint

    val trainData = sampleRdd.map(f => (new BDM(1, 1, f(::, 0).data), f(::, 1 to -1)))

    /*设置训练参数，建立模型*/
    // params:迭代步长，训练次数，交叉验证比例
    val params = Array(20.0, 20.0, 0.2)
    trainData.cache
    val numSamples = trainData.count
    println(s"numSamples = $numSamples.")
    val nnModel = new NeuralNet().
      setSize(Array(5, 7, 1)).
      setLayer(3).
      setActivation_function("sigm").
      setLearningRate(1.0).
      setScaling_learningRate(1.0).
      setWeightPenaltyL2(0.0).
      setNonSparsityPenalty(0.0).
      setSparsityTarget(0.05).
      setInputZeroMaskedFraction(0.0).
      setDropoutFraction(0.0).
      setOutput_function("sigm").
      NNTrain(trainData, params)

    //4 模型测试
    val NNForecast = nnModel.predict(trainData)
    val NNError = nnModel.Loss(NNForecast)
    println(s"NNError = $NNError.")
    val predictResult = NNForecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)

    val yMax = normMax(0, 0)
    val yMin = normMin(0, 0)
    for (i <- predictResult.indices) {
      val o1 = predictResult(i)._1
      val o2 = predictResult(i)._2
      val p1 = o1 * (yMax - yMin) + yMin
      val p2 = o2 * (yMax - yMin) + yMin
      println(s"实际值 $o1 ($p1); 预测值 $o2 ($p2)")
    }



    NNRunLog.logWeight(nnModel.weights, "更新后权重")


    NNRunLog.logTrainData(trainData, "原始样本数据")

    //*****************************例2（读取固定样本:来源于经典优化算法测试函数Sphere Model）*****************************//
    //    //2 读取样本数据,
    //    Logger.getRootLogger.setLevel(Level.WARN)
    //    val data_path = "/user/huangmeiling/deeplearn/data1"
    //    val examples = sc.textFile(data_path).cache()
    //    val train_d1 = examples.map { line =>
    //      val f1 = line.split("\t")
    //      val f = f1.map(f => f.toDouble)
    //      val id = f(0)
    //      val y = Array(f(1))
    //      val x = f.slice(2, f.length)
    //      (id, new BDM(1, y.length, y), new BDM(1, x.length, x))
    //    }
    //    val train_d = train_d1
    //    val params = Array(100.0, 20.0, 0.0)
    //    //3 设置训练参数，建立模型
    //    val NNmodel = new NeuralNet().
    //      setSize(Array(5, 7, 1)).
    //      setLayer(3).
    //      setActivation_function("tanh_opt").
    //      setLearningRate(2.0).
    //      setScaling_learningRate(1.0).
    //      setWeightPenaltyL2(0.0).
    //      setNonSparsityPenalty(0.0).
    //      setSparsityTarget(0.0).
    //      setOutput_function("sigm").
    //      NNtrain(train_d, params)
    //
    //    //4 模型测试
    //    val NNforecast = NNmodel.predict(train_d.map(f => (f._2, f._3)))
    //    val NNerror = NNmodel.Loss(NNforecast)
    //    println(s"NNerror = $NNerror.")
    //    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    //    println("预测结果——实际值：预测值：误差")
    //    for (i <- 0 until printf1.length)
    //      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))
    //    println("权重W{1}")
    //    val tmpw0 = NNmodel.weights(0)
    //    for (i <- 0 to tmpw0.rows - 1) {
    //      for (j <- 0 to tmpw0.cols - 1) {
    //        print(tmpw0(i, j) + "\t")
    //      }
    //      println()
    //    }
    //    println("权重W{2}")
    //    val tmpw1 = NNmodel.weights(1)
    //    for (i <- 0 to tmpw1.rows - 1) {
    //      for (j <- 0 to tmpw1.cols - 1) {
    //        print(tmpw1(i, j) + "\t")
    //      }
    //      println()
    //    }

    //*****************************例3（读取SparkMlib数据）*****************************//
    //例2 读取样本数据,转化：[y1,[x1 x2  x10]] => ([y1 y2],[x1 x2...x10])
    //    val data_path = "/data/sample_linear_regression_data.txt"
    //    val examples = MLUtils.loadLibSVMFile(sc, data_path).cache()
    //    val train_d1 = examples.map { f =>
    //      LabeledPoint(f.label, Vectors.dense(f.features.toArray))
    //    }
    //    val params = Array(100.0, 100.0, 0.0)
    //    val train_d = train_d1.map(f => (BDM((f.label, f.label * 0.5 + 2.0)), BDM(f.features.toArray)))
    //    val numExamples = train_d.count()
    //    println(s"numExamples = $numExamples.")

  }
}
