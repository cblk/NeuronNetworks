package neuron.test

import breeze.linalg.{*, max, sum, DenseMatrix => BDM, DenseVector => BDV}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by cblk on 2016/5/1.
  */
object Test {
  def main(args: Array[String]): Unit = {
    matrixTest()
  }

  def matrixTest(): Unit = {
    val m = BDM.ones[Double](6, 7)
    val n = m - m
    println(n)
    val normMax = sum(n(::, *)) / n.rows.toDouble
    println(normMax)

  }

  def arrayTest(): Unit = {
    val a = ArrayBuffer[Int](1, 2, 3)
    a += 4
    println(a(3))
  }
}
