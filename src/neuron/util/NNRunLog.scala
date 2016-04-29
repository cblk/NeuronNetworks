package neuron.util

import java.io.{File, FileOutputStream, PrintWriter}
import java.text.SimpleDateFormat

import breeze.linalg.{*, CSCMatrix => BSM, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV, axpy => brzAxpy, max => Bmax, min => Bmin, sum => Bsum, svd => brzSvd}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD

/**
  * Created by cblk on 2016/4/28.
  */
object NNRunLog {
  private val inputLogPath = "/usr/local/spark/worksapce/input.log"
  private val outputLogPath = "/usr/local/spark/worksapce/out.log"
  private val dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss")

  def logTrainData(trainData: RDD[(BDM[Double], BDM[Double])], tag: String): Unit = {
    val writer = getWriter(inputLogPath)

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
    val writer = getWriter(outputLogPath)

    writer.write("\r\n" + tag + "\t" + dateFormat.format(System.currentTimeMillis()) + "\r\n")

    writer.close()

  }

  def getWriter(path: String): PrintWriter = {
    val f = new File(path)
    if (!f.exists()) f.createNewFile()
    val o = new FileOutputStream(f, true)
    new PrintWriter(o, true)
  }

}
