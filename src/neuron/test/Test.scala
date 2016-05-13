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
    val data = Array(Array(1, 2, 3), Array(4, 5, 6))
    val m = new BDM[Int](3, 2, Array(1, 2, 3, 4, 5, 6)).t
    val v = BDV(0, 0)
    val n = m(0, ::).t
    println(n)

  }

  def arrayTest(): Unit = {
    val a = ArrayBuffer[Int](1, 2, 3)
    a += 4
    println(a(3))
  }
}
