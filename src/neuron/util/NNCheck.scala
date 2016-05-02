package neuron.util

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import neuron.spark.{NNConfig, NNLabel, NeuralNet}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by cblk on 2016/5/1.
  */
object NNCheck {
  /**
    * 单个样本代价函数
 *
    * @param a0      : 特征向量
    * @param label  : 输出
    * @param nn_W   : 神经网络权重
    * @param config : 神经网络结构
    * @return 代价输出
    */
  def lossFun(
               a0: BDV[Double],
               label: Double,
               nn_W: Array[BDM[Double]],
               config: NNConfig
             ): Double = {
    val A = ArrayBuffer[BDV[Double]]()
    A += a0
    val dropOutMask = ArrayBuffer[BDV[Double]]()
    dropOutMask += new BDV[Double](Array(0.0))
    for (j <- 1 until config.layer - 1) {
      // 计算每层输出
      // nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}')
      // nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');

      //Ai为前一层的输出，是一个向量
      val Ai = A(j - 1)
      //Wi为当前层与前一层之间的权重矩阵
      val Wi = nn_W(j - 1)
      //Zj是当前层的
      val Zj = Wi * Ai
      val Aj = config.activation_function match {
        case "sigm" =>
          NeuralNet.sigm(Zj)
        case "tanh_opt" =>
          NeuralNet.tanh_opt(Zj)
      }
      // dropout计算
      // Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分
      // 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
      // 参照 http://www.cnblogs.com/tornadomeet/p/3258122.html
      val dropoutAj = if (config.dropoutFraction > 0) {
        if (config.testing == 1) {
          val Aj2 = Aj * (1.0 - config.dropoutFraction)
          Array(new BDV[Double](Array(0.0)), Aj2)
        } else {
          NeuralNet.DropoutWeight(Aj, config.dropoutFraction)
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
    val Al_ = A(config.layer - 2)
    val Wl_ = nn_W(config.layer - 2)
    val Zl: BDV[Double] = Wl_ * Al_
    val Al = config.output_function match {
      case "sigm" =>
        NeuralNet.sigm(Zl)
      case "linear" =>
        Zl
    }
    0.5 * (Al(0) - label) * (Al(0) - label)
  }

  /**
    * 梯度检验（针对单个样本）
    **/
  def checkGradient(
                     a0: BDV[Double],
                     label: Double,
                     config: NNConfig,
                     layer: Int,
                     nn_W: Array[BDM[Double]]
                   ): BDM[Double] = {
    val rows = config.size(layer + 1)
    val cols = config.size(layer) + 1
    val Wi = nn_W(layer)
    val d = new ArrayBuffer[Double](rows * cols)
    for (j <- 0 until cols) {
      for (i <- 0 until rows) {
        val base = getBase(rows, cols, i, j, layer, nn_W)
        val plus = lossFun(a0, label, base._1, config)
        val minus = lossFun(a0, label, base._2, config)
        val gradient = (plus - minus)/0.0002
        d += gradient
      }
    }
    new BDM[Double](rows, cols, d.toArray)
  }

  def getBase(rows: Int, cols: Int,
              i: Int, j: Int,
              layer: Int, nn_W: Array[BDM[Double]]):(Array[BDM[Double]], Array[BDM[Double]]) = {
    val delta = BDM.zeros[Double](rows, cols)
    delta(i, j) += 0.0001
    val nn_WPlus = copyBDM(nn_W)
    nn_WPlus(layer) = nn_W(layer) + delta
    val nn_WMinus = copyBDM(nn_W)
    nn_WMinus(layer) = nn_W(layer) - delta
    (nn_WPlus, nn_WMinus)
  }

  def checkGradient(bpResult: RDD[(NNLabel, Array[BDM[Double]])], config: NNConfig, nn_W: Array[BDM[Double]]): Unit = {
    val res = bpResult.collect()
    for (i <- res.indices) {
      val nnLabel = res(i)._1
      val dW = res(i)._2
      for (j <- 0 until config.layer - 1) {
        val check = checkGradient(nnLabel.A(0), nnLabel.label, config, j, nn_W)
        NNRunLog.logGradient(check, dW(j), j)

      }

    }
  }

  def copyBDM(src : Array[BDM[Double]]): Array[BDM[Double]] = {
    val dst = new Array[BDM[Double]](src.length)
    for (i <- src.indices) {
      val srcMatrix = src(i)
      val dstMatrix = new BDM[Double](srcMatrix.rows, srcMatrix.cols, srcMatrix.data)
      dst(i) = dstMatrix
    }
    dst
  }

}
