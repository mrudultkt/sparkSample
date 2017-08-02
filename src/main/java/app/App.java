package app;


import util.CosineSimilarity;
import util.KMeansClustering;
import util.TFIDF;
import util.Word2VecUtil;

public class App {
    public static void main(String[] args) {
      /* Word2VecUtil word2VecUtil = new Word2VecUtil();
       word2VecUtil.word2Vec();*/

       /* CosineSimilarity cosineSimilarity = new CosineSimilarity();
        double cosineSimilarityScore = cosineSimilarity.cosineSimilarityScore("Jack is good",
                "Julie loves me more than Linda loves me");
        System.out.println(cosineSimilarityScore);*/

       /* TFIDF tfidf = new TFIDF();
        tfidf.tFIDF();*/

        KMeansClustering kMeansClustering = new KMeansClustering();
        kMeansClustering.word2VecKMeans();
    }

}
