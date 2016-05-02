package neuron.util

import breeze.linalg.{*, max, min, sum, DenseMatrix => BDM, DenseVector => BDV}

/**
  * Created by cblk on 2016/5/2.
  */
object Norm {
  def apply(sampleMatrix: BDM[Double]): BDM[Double] = {
    val rows = sampleMatrix.rows

    val feature = sampleMatrix(::, 1 to -1)
    val mean = sum(feature(::, *)) / rows.toDouble
    val normMean = BDM.ones[Double](rows, 1) * mean

    val diff = feature - normMean
    val square = diff :* diff
    val variance = sum(square(::, *)) / (rows.toDouble * 2.0)


    val normVar = BDM.ones[Double](rows, 1) * variance
    val norm1 = diff :/ normVar

    val label = new BDM[Double](rows, 1, sampleMatrix(::, 0).data)
    val normMax = max(label)
    val normMin = min(label)
    val norm2 = (label - normMin) / (normMax - normMin)
    BDM.horzcat(norm2, norm1)
  }
}
