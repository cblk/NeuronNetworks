package neuron.util

import java.io.{File, FileOutputStream, PrintWriter}
import java.text.SimpleDateFormat

import breeze.linalg.{CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, max => Bmax, min => Bmin, sum => Bsum, svd => brzSvd}
import neuron.spark.NNLabel
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
  private val gradientCheckPath = "C:\\Users\\cblk\\Desktop\\test\\gradient.log"

  private val dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss")
  val isLog = false

  def logTrainData(trainData: RDD[(Double, BDV[Double])], tag: String): Unit = {
    if (true) {
      val writer = getWriter(inputLogPath, true)

      writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")
      val y_x = trainData.map(f => (f._1, f._2.toArray)).collect().map { f => ((new ArrayBuffer[Double]() :+ f._1) ++ f._2).toArray }
      for (i <- y_x.indices) {
        for (j <- y_x(i).indices) {
          writer.write(y_x(i)(j) + "\t")
        }
        writer.write("\r\n")
      }

      writer.close()
    }
  }

  def logAn(label: RDD[NNLabel], tag: String): Unit = {
    if (isLog) {
      val writer = getWriter(outputLogPath, true)
      writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")

      val nnLabels = label.collect()
      for (i <- nnLabels.indices) {
        val A = nnLabels(i).A
        val l = nnLabels(i).label
        writer.write(s"第 $i 个样本的各层输出如下：\r\n")
        for (j <- A.indices) {
          writer.write(s"第 $j 层:")
          for (k <- 0 until A(j).length) {
            writer.write("\t" + A(j)(k))
          }
          writer.write("\r\n")
        }
        writer.write("label:" + l + "\r\n")

      }

      writer.close()
    }


  }

  def logWeight(weights: Array[BDM[Double]], tag: String): Unit = {
    if (isLog) {
      val writer = getWriter(weightPath, true)
      writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")
      for (i <- weights.indices) {
        val w0 = weights(i)
        writer.write("W" + i + ";rows:" + w0.rows + ";cols:" + w0.cols + "\r\n")
        for (i <- 0 until w0.rows) {
          for (j <- 0 until w0.cols) {
            writer.write(w0(i, j) + "\t")
          }
          writer.write("\r\n")
        }
      }
      writer.close()
    }

  }

  def logGradient(check: BDM[Double], target: BDM[Double], layer: Int): Unit = {
    if (isLog) {
      val writer = getWriter(gradientCheckPath, true)
      writer.write("\r\n" + s"The $layer layer gradient check" + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")
      logBDM(check, writer, "估计值")
      logBDM(target, writer, "实际值")
      logBDM(check :/ target, writer, "比值")
      writer.close()
    }

  }

  def logBDM(m: BDM[Double], writer: PrintWriter, tag: String): Unit = {
    writer.write(tag + "\t" + "rows:" + m.rows + ";cols:" + m.cols + "\r\n")
    for (i <- 0 until m.rows) {
      for (j <- 0 until m.cols) {
        writer.write(m(i, j) + "\t")
      }
      writer.write("\r\n")
    }
  }

  def getWriter(path: String, isAppend: Boolean): PrintWriter = {
    val f = new File(path)
    if (!f.exists()) f.createNewFile()
    val o = new FileOutputStream(f, isAppend)
    new PrintWriter(o, true)
  }

}
