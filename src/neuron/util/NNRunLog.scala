package neuron.util

import java.io.{File, FileOutputStream, PrintWriter}
import java.text.SimpleDateFormat

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, max => Bmax, min => Bmin, sum => Bsum, svd => brzSvd}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by cblk on 2016/4/28.
  */
object NNRunLog {
/*  private val inputLogPath = "/usr/local/spark/workspace/input.log"
  private val outputLogPath = "/usr/local/spark/workspace/out.log"
  private val weightPath = "/usr/local/spark/workspace/weight.log"*/

  private val inputLogPath = "C:\\Users\\cblk\\Desktop\\test\\input.log"
  private val outputLogPath = "C:\\Users\\cblk\\Desktop\\test\\out.log"
  private val weightPath = "C:\\Users\\cblk\\Desktop\\test\\weight.log"

  private val dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss")

  def logTrainData(trainData: RDD[(BDM[Double], BDM[Double])], tag: String): Unit = {
    val writer = getWriter(inputLogPath, false)

    writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")
    val y_x = trainData.map(f => (f._1.toArray, f._2.toArray)).collect().map { f => ((new ArrayBuffer() ++ f._1) ++ f._2).toArray }
    for (i <- y_x.indices) {
      for (j <- y_x(i).indices) {
        writer.write(y_x(i)(j) + "\t")
      }
      writer.write("\r\n")
    }

    writer.close()
  }

  def logAn(out: BDM[Double], tag: String): Unit = {
    val writer = getWriter(outputLogPath, true)

    writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")

    writer.close()

  }

  def logWeight(weights: Array[BDM[Double]], tag: String): Unit = {
    val writer = getWriter(weightPath, true)
    writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")
    for (i <- weights.indices) {
      val w0 = weights(i)
      writer.write("W" + i + ";rows:" + w0.rows + "cols" + w0.cols + "\r\n")
      for (i <- 0 until w0.rows) {
        for (j <- 0 until w0.cols) {
          writer.write(w0(i, j) + "\t")
        }
        writer.write("\r\n")
      }
    }

    writer.close()
  }

  def getWriter(path: String, isAppend: Boolean): PrintWriter = {
    val f = new File(path)
    if (!f.exists()) f.createNewFile()
    val o = new FileOutputStream(f, isAppend)
    new PrintWriter(o, true)
  }

}
