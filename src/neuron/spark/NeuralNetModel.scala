package neuron.spark

/**
  * Created by mao on 2016/4/16 0016.
  */
import breeze.linalg.{sum, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import org.apache.spark.rdd.RDD

/**
  * label：目标矩阵
  * features：特征矩阵
  * predict_label：预测矩阵
  * error：误差
  */
case class PredictResult(label: Double, features: BDV[Double], predict_label: Double, error: Double) extends Serializable

/**
  * NN(neural network)
  */

class NeuralNetModel(
                      val config: NNConfig,
                      val weights: Array[BDM[Double]]) extends Serializable {

  /**
    * 返回预测结果
    *  返回格式：(label, feature,  predict_label, error)
    */
  def predict(dataMatrix: RDD[(Double, BDV[Double])]): RDD[PredictResult] = {
    val sc = dataMatrix.sparkContext
    val bc_nn_W = sc.broadcast(weights)
    val bc_config = sc.broadcast(config)

    val ffResult = NeuralNet.NNff(dataMatrix, bc_config, bc_nn_W)
    val predict = ffResult.map { f =>
      val label = f._1.label
      val error = f._1.error
      val An = f._1.A(bc_config.value.layer - 1)
      val A1 = f._1.A(0)(1 to -1)
      PredictResult(label, A1, An(0), error)
    }
    predict
  }

  /**
    * 计算输出误差
    * 平均误差;
    */
  def Loss(predict: RDD[PredictResult]): Double = {
    val predict1 = predict.map(f => f.error)
    // error and loss
    // 输出误差计算
    val loss1 = predict1
    val (loss2, count) = loss1.treeAggregate((0.0, 0L))(
      seqOp = (c, v) => {
        // c: (e, count), v: (m)
        val e1 = c._1
        val e2 = v * v
        val esum = e1 + e2
        (esum, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (e, count)
        val e1 = c1._1
        val e2 = c2._1
        val esum = e1 + e2
        (esum, c1._2 + c2._2)
      })
    val Loss = loss2 / count.toDouble
    Loss * 0.5
  }

}
