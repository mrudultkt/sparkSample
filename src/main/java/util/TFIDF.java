package util;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.linalg.Vector;

import java.util.Arrays;
import java.util.List;

public class TFIDF {
    private SparkConf conf;
    private JavaSparkContext sc;
   // JavaRDD<String> linesOfFile;

    public TFIDF(){
        conf = new SparkConf().setMaster("local[2]").setAppName("word2vecapp");;
        sc =  new JavaSparkContext(conf);
       // linesOfFile = sc.textFile("data.txt");
    }

    public void tFIDF(){
        HashingTF tf = new HashingTF();
        JavaRDD<List<String>> documents = sc.parallelize(Arrays.asList(
                Arrays.asList("this is a sentence sentence".split(" ")),
                Arrays.asList("this is another sentence".split(" ")),
                Arrays.asList("this is still a sentence".split(" "))), 2);
        System.out.println("#############");
        System.out.println(documents.collect().toString());
        JavaRDD<Vector> termFreqs = tf.transform(documents);
        List<Vector> tfVectors = termFreqs.collect();
        tfVectors.forEach(v->{
            System.out.println(v.toBreeze());
        });
        System.out.println("*******");
        System.out.println(termFreqs.collect().toString());

        IDF idf = new IDF();
        JavaRDD<Vector> tfIdfs = idf.fit(termFreqs).transform(termFreqs);
        List<Vector> localTfIdfs = tfIdfs.collect();
        int indexOfThis = tf.indexOf("this");
        int indexOfA = tf.indexOf("a");
        int indexOfStill = tf.indexOf("still");
        System.out.println("indexOfThis" + indexOfThis);
        System.out.println("indexOfA" + indexOfA);
        System.out.println("indexOfStill" + indexOfStill);
        for (Vector v: localTfIdfs) {
            System.out.println(v.toString());
            System.out.println(v.toBreeze());
        }
    }
}
