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
import neuron.util.{NNRunLog, Norm, RandomSampleData}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by mao on 2016/4/16 0016.
  */
object TestForNN {
  def main(args: Array[String]) {

    /*构建Spark对象*/
    val conf = new SparkConf().setAppName("NNtest")
      .setMaster("spark://spark4:7077")
      .setJars(List("C:\\Users\\cblk\\IdeaProjects\\NeuronNetworks\\out\\artifacts\\NeuronNetworks_jar\\NeuronNetworks.jar"))
    val sc = new SparkContext(conf)
    //val nameNode = "hdfs://10.141.211.123:9000/"

    /*随机生成测试数据*/
    // 生成原始随机样本数据
    Logger.getRootLogger.setLevel(Level.ERROR)
    val sampleNum = 10
    val xDimens = 2
    val sampleMatrix = RandomSampleData.RandM(sampleNum, xDimens, -10, 10, "xor")
    // 归一化

    val norm2 = Norm(sampleMatrix)

    // 将矩阵形式的原始样本拆分成数组形式保存
    val sampleArray = ArrayBuffer[BDV[Double]]()
    for (i <- 0 until sampleNum) {
      val mi = sampleMatrix(i, ::).inner
      sampleArray += mi
    }

    //由本地的样本数据集合生成分布式内存数据集Rdd
    val sampleRdd = sc.parallelize(DataProcess.getSample(sc, "1387880"), 1)
    //sc.setCheckpointDir(nameNode + "checkpoint")
    //sampleRdd.checkpoint

    val trainData = sampleRdd.map(f => (f(0), f(1 to -1)))
    NNRunLog.logTrainData(trainData, "原始样本数据")

    /*设置训练参数，建立模型*/
    // params:迭代步长，训练次数，交叉验证比例
    val params = Array(100, 2000, 0.2)
    trainData.cache
    val numSamples = trainData.count
    println(s"numSamples = $numSamples.")
    val nnModel = new NeuralNet().
      setSize(Array(6, 5, 1)).
      setLayer(3).
      setActivation_function("sigm").
      setLearningRate(0.2).
      setMomentum(0.0).
      setScaling_learningRate(1.0).
      setWeightPenaltyL2(0.02).
      setNonSparsityPenalty(0.0).
      setSparsityTarget(0.05).
      setInputZeroMaskedFraction(0.1).
      setDropoutFraction(0.0).
      setOutput_function("sigm").
      NNTrain(trainData, params)

    //4 模型测试
    val NNForecast = nnModel.predict(trainData)
    val NNError = nnModel.Loss(NNForecast)
    println(s"NNError = $NNError.")
    val predictResult = NNForecast.map(f => (f.label, f.predict_label)).take(200)


    for (i <- predictResult.indices) {
      val o1 = predictResult(i)._1
      val o2 = predictResult(i)._2

      println(s"实际值 $o1 ; 预测值 $o2 ")
    }
    NNRunLog.logWeight(nnModel.weights, "更新后权重")


  }
}
