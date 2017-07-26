package util;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.util.Arrays;

public class Word2VecUtil {

    private SparkConf conf;
    private JavaSparkContext sc;
    JavaRDD<String> linesOfFile;

    public Word2VecUtil() {
        conf = new SparkConf().setMaster("local[2]").setAppName("word2vecapp");;
        sc =  new JavaSparkContext(conf);
        linesOfFile = sc.textFile("data.txt");
    }

    public void word2Vec(){

        JavaRDD<Iterable<String>> words = linesOfFile.map(s -> Arrays.asList(s.split(" ")));
        Word2Vec word2Vec = new Word2Vec();
        word2Vec.setVectorSize(3);
        word2Vec.setMinCount(0);
        Word2VecModel model = word2Vec.fit(words);
        System.out.println("********** Individual vectors");
        System.out.println(model.getVectors());
        String text = "I heard about Spark";
        String[] wordArray = text.split(" ");
        Vector resultVector1 = null;
        for (String word : wordArray) {
            if(resultVector1 != null){
                resultVector1 = addVectors(resultVector1, model.transform(word));
            }else{
                resultVector1 = model.transform(word);
            }
        }
        System.out.println("result vector 1 : " + resultVector1.toString() );

        text = "could use case classes";
        wordArray = text.split(" ");
        Vector resultVector2 = null;
        for (String word : wordArray) {
            if(resultVector2 != null){
                resultVector2 = addVectors(resultVector2, model.transform(word));
            }else{
                resultVector2 = model.transform(word);
            }
        }
        System.out.println("result vector 2 : " + resultVector2.toString());

        calculateDistance(resultVector1.toArray(), resultVector2.toArray());
        System.out.println("#######  " + Vectors.sqdist(resultVector1, resultVector2));


    }

    public Vector addVectors(Vector v1, Vector v2){
        Vector sum = null;
        double[] v1Array = v1.toArray();
        double[] v2Array = v2.toArray();
        double[] v3Array= { 0 , 0, 0 };
        // double[] v3Array = new double[v1Array.length];
        for(int i = 0; i < v1Array.length; i++){
            v3Array[i] = v1Array[i] + v2Array[i];
        }
        sum = Vectors.dense(v3Array);
        return sum;
    }

    // to calculate Euclidean distance
    public static double calculateDistance(double[] array1, double[] array2)
    {
        double Sum = 0.0;
        for(int i=0;i<array1.length;i++) {
            Sum = Sum + Math.pow((array1[i]-array2[i]),2.0);
        }
        System.out.println("Distance b/w 2 vectors :" + Math.sqrt(Sum));
        return Math.sqrt(Sum);
    }

}
