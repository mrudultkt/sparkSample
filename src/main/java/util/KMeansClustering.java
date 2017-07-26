package util;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class KMeansClustering {
    private SparkConf conf;
    private JavaSparkContext sc;
    JavaRDD<String> linesOfFile;

    public KMeansClustering(){
        conf = new SparkConf().setMaster("local[2]").setAppName("word2vecapp");;
        sc =  new JavaSparkContext(conf);
        linesOfFile = sc.textFile("kmeansfile.txt");
    }

    public void kMeans(){

        JavaRDD<Vector> parsedData = linesOfFile.map(s->{
            String[] sarray = s.split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                values[i] = Double.parseDouble(sarray[i]);
            }
            return Vectors.dense(values);
        });
        System.out.println("par" + parsedData.collect());
        parsedData.cache();
        // Cluster the data into two classes using KMeans
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
    }
}
