package neuron.test

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import neuron.util.NNRunLog
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Dataset, SQLContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by cblk on 2016/4/30.
  */

case class Record(id: String, barCode: String, var date: String, total: Option[Double], fee1: Option[Double], fee2: Option[Double])

case class CleanRecord(id: String, date: Int, total: Double, fee1: Double, fee2: Double)

object DataProcess {
  val nameNode = "hdfs://10.141.211.123:9000/"

  private val fileList = Array("201201.csv", "201202.csv", "201203.csv", "201204.csv", "201205.csv", "201206.csv",
    "201207.csv", "201208.csv", "201209.csv", "201210.csv", "201211.csv", "201212.csv")
  private val days = Array(31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  private val accDays = Array(0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

  private def getRecords(sc: SparkContext): Array[Dataset[Array[CleanRecord]]] = {
    val sqlContext = new SQLContext(sc)
    val recordDSList: Array[Dataset[Array[CleanRecord]]] = new Array(12)
    import sqlContext.implicits._

    for (i <- 1 to 12) {
      val df = sqlContext.read
        .format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .option("inferSchema", "true") // Automatically infer data types
        .load(nameNode + "data/" + fileList(i - 1))
      val ds = df.as[Record]
      val cleanRecord = ds.groupBy(r => r.id)
        .mapGroups((key, records) => records.toArray)
        .filter(r => r.length == days(i - 1) && !checkNull(r))
      val data = cleanRecord.map { r =>
        val records = r.sortWith((r1, r2) => r1.total.getOrElse(0.0) < r2.total.getOrElse(0.0))
        val newRecords = new ArrayBuffer[CleanRecord]()
        var fee1: Double = 0.0
        var fee2: Double = 0.0
        var total: Double = 0.0
        for (j <- records.indices) {
          if (j != 0) {
            fee1 = records(j).fee1.getOrElse(0.0) - fee1
            fee2 = records(j).fee2.getOrElse(0.0) - fee2
            total = records(j).total.getOrElse(0.0) - total
          }

          newRecords += new CleanRecord(records(j).id, accDays(i - 1) + j, total, fee1, fee2)
          fee1 = records(j).fee1.getOrElse(0.0)
          fee2 = records(j).fee2.getOrElse(0.0)
          total = records(j).total.getOrElse(0.0)
        }
        newRecords.toArray
      }
      recordDSList(i - 1) = data
    }
    recordDSList
  }

  private def checkNull(records: Array[Record]): Boolean = {
    for (i <- records.indices) {
      if (checkNull(records(i))) true
    }
    false
  }

  private def checkNull(record: Record): Boolean = {
    if (record.id == null || record.fee1.isEmpty || record.fee2.isEmpty || record.total.isEmpty) true
    else false
  }


  def getSample(sc: SparkContext, id: String): Array[BDV[Double]] = {
    val sample = NNRunLog.getBDV(id)
    if (sample.nonEmpty) {
      sample.toArray
    } else {
      val recordDSList = getRecords(sc)
      val res = recordDSList.flatMap { rl =>
        rl.filter(r => r(0).id == id).collect()
      }

      for (i <- res.indices) {
        for (j <- res(i).indices) {
          val rs = res(i)
          if (j == 0) {
          } else if (j < 3) {
            sample += new BDV(Array(rs(j).fee1, rs(j + 1).fee1, rs(j + 2).fee1, rs(j + 3).fee1, rs(j + 4).fee1, rs(j + 5).fee1, rs(j + 6).fee1))
          } else if (j >= res(i).length - 3) {
            sample += new BDV(Array(rs(j).fee1, rs(j - 1).fee1, rs(j - 2).fee1, rs(j - 3).fee1, rs(j - 4).fee1, rs(j - 5).fee1, rs(j - 6).fee1))
          } else {
            sample += new BDV(Array(rs(j).fee1, rs(j - 3).fee1, rs(j - 2).fee1, rs(j - 1).fee1, rs(j + 1).fee1, rs(j + 2).fee1, rs(j + 3).fee1))
          }
        }
      }
      NNRunLog.saveBDV(id, sample.toArray)
      sample.toArray
    }
  }

  def arrayToMatrix(data: Array[BDV[Double]]): BDM[Double] = {
   new BDM(data(0).length, data.length, data.flatMap(v => v.data)).t
  }

  def matrixToArray(data: BDM[Double]): Array[BDV[Double]] = {
    val d = new ArrayBuffer[BDV[Double]]()
    for (i <- 0 until data.rows) {
      val s = data(i, ::).t
      d += s
    }
    d.toArray
  }
}
