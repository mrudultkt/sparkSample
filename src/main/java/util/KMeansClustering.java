package util;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import java.util.Arrays;
import java.util.List;

public class KMeansClustering {
    private SparkConf conf;
    private JavaSparkContext sc;
    private JavaRDD<String> clusteringData;
    private JavaRDD<String> word2vecData;

    public KMeansClustering(){
        conf = new SparkConf().setMaster("local[2]").setAppName("word2vecapp");
        sc =  new JavaSparkContext(conf);
        clusteringData = sc.textFile("kmeansfile.txt");
        word2vecData = sc.textFile("data.txt");
    }

    public void kMeans(){

        JavaRDD<Vector> parsedData = clusteringData.map(s->{
            String[] sarray = s.split(" ");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                values[i] = Double.parseDouble(sarray[i]);
            }
            return Vectors.dense(values);
        });
        System.out.println("parsed data" + parsedData.collect());
        parsedData.cache();
        // Cluster the data into two classes using KMeans
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);
        Vector[] clusterCenters = clusters.clusterCenters();
        System.out.println("cluster centres" );
        for(int i = 0; i < clusterCenters.length; i++){
            System.out.println(clusterCenters[i].toBreeze());
        }
        System.out.println("value of k :" + clusters.k());
        System.out.println("Predicted value :" + clusters.predict(Vectors.dense(new double[] {25, 12, 56, 23 ,1, 3, 1})));
        //System.out.println("" + clusters.predict([]));
        // Evaluate clustering by computing Within Set Sum of Squared Errors (sum of squared distances of points to their nearest center)
        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
    }

    public void word2VecKMeans() {

        JavaRDD<Iterable<String>> words = word2vecData.map(s -> Arrays.asList(s.split(" ")));
        Word2Vec word2Vec = new Word2Vec();
        word2Vec.setVectorSize(3);
        word2Vec.setMinCount(0);
        Word2VecModel model = word2Vec.fit(words);
        System.out.println("********** Individual vectors **********");
        System.out.println(model.getVectors());
        String text = "I heard about Spark I wish I could use";
        List<String> textWords = Arrays.asList(text.split(" "));
        JavaRDD<String> textWordsRDD = sc.parallelize(textWords);
        JavaRDD<Vector> vectorJavaRDD = textWordsRDD.map(s -> {
          return   model.transform(s);
        });
        int numClusters = 2;
        int numIterations = 20;
        KMeansModel kMeansModel  = KMeans.train(vectorJavaRDD.rdd(), numClusters, numIterations);
        System.out.println(kMeansModel.computeCost(vectorJavaRDD.rdd()));

    }
}
