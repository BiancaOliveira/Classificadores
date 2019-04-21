package Banana;

import java.io.*;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.core.FastVector;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.Bagging;
import weka.filters.supervised.instance.Resample;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.gui.beans.TestSetEvent;
import weka.gui.beans.TrainTestSplitMaker;
import weka.gui.beans.TrainingSetEvent;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.util.Random;
import static javafx.scene.input.KeyCode.F;
import static javafx.scene.input.KeyCode.V;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.Bagging;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import libsvm.svm;
import weka.classifiers.functions.pace.DiscreteFunction;
import weka.classifiers.functions.supportVector.PolyKernel;
import static weka.classifiers.lazy.IBk.WEIGHT_INVERSE;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;
import static weka.classifiers.lazy.IBk.WEIGHT_SIMILARITY;
import weka.core.EuclideanDistance;
import weka.core.SelectedTag;
/**
 *
 * @author Abrun
 */
public class Banana {
   
    /**
     * @param args the command line arguments
     */
    public static int acharK(Instances valida, Instances teste) throws Exception{
        int melhorK = 0;
        double aux = 0;

        for(int k = 1; k<16; k+=2){

            IBk ClassificadorKnn = new IBk(k);
            ClassificadorKnn.buildClassifier(valida);
            Evaluation eval2 = new Evaluation(valida);
            eval2.evaluateModel(ClassificadorKnn, teste);                
//            System.out.println("KNN: "+String.valueOf(eval2.pctCorrect()/100)+"\n");
            if(aux < eval2.pctCorrect()/100){
                aux = eval2.pctCorrect()/100;
                melhorK = k;
            }
        }
        return melhorK;
    }
    public static double melhorC(Instances valida, Instances teste) throws Exception{
        double melhorC = 0;
        double aux = 0;

        for(int i = 1; i<10; i++){
            SMO Classificador5 = new SMO();
            RBFKernel rbf = new RBFKernel();
            Classificador5.setC(i);
            Classificador5.buildClassifier(valida);
            Evaluation eval5 = new Evaluation(valida);
            eval5.evaluateModel(Classificador5, teste);                
             if(aux < eval5.pctCorrect()/100){
                aux = eval5.pctCorrect()/100;
                melhorC = i;
            }
        }
        return melhorC;

    }
    
    public static int melhorTT(Instances valida, Instances teste) throws Exception{
        int melhorTT = 0;
        double aux = 0;

        for(int i = 500; i<1500; i+=100){
             MultilayerPerceptron ClassificadorMLP  = new MultilayerPerceptron();        
            ClassificadorMLP.setLearningRate(0.1);
            ClassificadorMLP.setMomentum(0.1);
            ClassificadorMLP.setTrainingTime(i); 
            ClassificadorMLP.setHiddenLayers("1");    //Com mais de 5 camadas piora o resultado            
            ClassificadorMLP.buildClassifier(valida);
            Evaluation eval6 = new Evaluation(valida);
            eval6.evaluateModel(ClassificadorMLP, teste);                
             if(aux < eval6.pctCorrect()/100){
                aux = eval6.pctCorrect()/100;
                melhorTT = i;
            }
        }
        return melhorTT;

    }
   
    public static void decidindoMelhoresParametros(int x) throws IOException, Exception{
          //carrega os dados de treino 
     double mediaMLP = 0, mediaSMO_Poly = 0, mediaSMO_RBK = 0,  mediaAD_SP = 0, mediaAD_CP = 0;
     double mediaKNN1 = 0, mediaKNN2 = 0, mediaKNN3 = 0;
     double C = 0;
     int k = 0, TT = 0;

     for(int i = 0; i< 20; i++){
        BufferedReader  breader = null;    
        breader = new BufferedReader(new FileReader("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\Banana.arff"));    
        Instances todos = new Instances (breader);
        todos.setClassIndex(todos.numAttributes()-1); //identifica que a classe está na última coluna

        //faz a separação em 50% para treino e 50% para o restante (temp)      
        StratifiedRemoveFolds fold = new StratifiedRemoveFolds();
        fold.setInputFormat( todos );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 1 );
        Instances treino = Filter.useFilter( todos, fold );
        
        //depois de separado, o conjunto de treino é gravado em disco em formato .arff
        FileWriter arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaTreino.arff"); 
        PrintWriter gravarArq = new PrintWriter(arq);
        String Temp = treino.toString();
        gravarArq.printf(Temp);
        arq.close();    
        
        fold = new StratifiedRemoveFolds();
        fold.setInputFormat( todos );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 2 );        
        Instances temporario = Filter.useFilter( todos, fold );   
        
        //a segunda metade da base original é dividida em teste e validação
        fold.setInputFormat( temporario );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 1 );        
        Instances valida = Filter.useFilter(temporario, fold );
        
        //o conjunto de validação é gravado em disco em formato .arff     
        arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaValidacao.arff"); 
        gravarArq = new PrintWriter(arq);
        Temp = valida.toString();
        gravarArq.printf(Temp);
        arq.close();     
                
        fold.setInputFormat( temporario );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 2 );        
        Instances teste = Filter.useFilter( temporario, fold );
        
        //o conjunto de teste é gravado em disco em formato .arff     
        arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaTeste.arff"); 
        gravarArq = new PrintWriter(arq);
        Temp = teste.toString();
        gravarArq.printf(Temp);
        arq.close();            
        
        if(i==0){
            k = acharK(valida,teste);
            C = melhorC(valida,teste);
            TT = melhorTT(valida,teste);
            
            System.out.println("K = "+k);
            System.out.println("C = "+C);
            System.out.println("TT = "+TT);

           
        }
        
        //Classificadores
       
      //  Descobrindo os melhores parametros com o conjunto validação
        MultilayerPerceptron ClassificadorMLP  = new MultilayerPerceptron();        
        ClassificadorMLP.setLearningRate(0.1);
        ClassificadorMLP.setMomentum(0.1);
        ClassificadorMLP.setTrainingTime(TT); 
        ClassificadorMLP.setHiddenLayers("5");    //Com mais de 5 camadas piora o resultado            
        ClassificadorMLP.buildClassifier(valida);
        Evaluation eval6 = new Evaluation(valida);
        eval6.evaluateModel(ClassificadorMLP, teste);                
        gravarArq.printf(String.valueOf(eval6.pctCorrect()/100)+"\n"); 
       // System.out.println("MLP: "+String.valueOf(eval6.pctCorrect()/100)+"\n");         
        mediaMLP = mediaMLP + eval6.pctCorrect()/100;

      
        
        SMO ClassificadorSMOP_RBFKernel = new SMO();
        RBFKernel rbf = new RBFKernel(); // piora a mediaSMO
        ClassificadorSMOP_RBFKernel.setC(C);
        ClassificadorSMOP_RBFKernel.setKernel(rbf);
        ClassificadorSMOP_RBFKernel.buildClassifier(valida);
        Evaluation eval5 = new Evaluation(valida);
        eval5.evaluateModel(ClassificadorSMOP_RBFKernel, teste);                
        gravarArq.printf(String.valueOf(eval5.pctCorrect()/100)+"\n"); 
//        System.out.println("SVM: "+String.valueOf(eval5.pctCorrect()/100)+"\n"); 
        mediaSMO_RBK = mediaSMO_RBK + eval5.pctCorrect()/100;
        
        
        SMO ClassificadorSMOP_KerPolynel = new SMO();
        PolyKernel pk = new PolyKernel();
        ClassificadorSMOP_KerPolynel.setC(C);
        ClassificadorSMOP_KerPolynel.setKernel(pk);
        ClassificadorSMOP_KerPolynel.buildClassifier(valida);
        Evaluation eval1 = new Evaluation(valida);
        eval1.evaluateModel(ClassificadorSMOP_KerPolynel, teste);                
        gravarArq.printf(String.valueOf(eval1.pctCorrect()/100)+"\n"); 
//        System.out.println("SVM: "+String.valueOf(eval5.pctCorrect()/100)+"\n"); 
        mediaSMO_Poly = mediaSMO_Poly + eval1.pctCorrect()/100;
        
        
        IBk ClassificadorKnn1 = new IBk(WEIGHT_NONE); //todos as metricas de distancia obtem a mesma media
        ClassificadorKnn1.setKNN(k);
        ClassificadorKnn1.buildClassifier(valida);
        Evaluation eval2 = new Evaluation(valida);
        eval2.evaluateModel(ClassificadorKnn1, teste);                
        gravarArq.printf(String.valueOf(eval2.pctCorrect()/100)+"\n"); 
        //System.out.println("KNN "+ i +" : "+String.valueOf(eval2.pctCorrect()/100)+"\n");   
        mediaKNN1 = mediaKNN1 + eval2.pctCorrect()/100;
        
        IBk ClassificadorKnn2 = new IBk(WEIGHT_INVERSE); //todos as metricas de distancia obtem a mesma media
        ClassificadorKnn2.setKNN(k);
        ClassificadorKnn2.buildClassifier(valida);
        Evaluation eva2 = new Evaluation(valida);
        eva2.evaluateModel(ClassificadorKnn2, teste);                
        gravarArq.printf(String.valueOf(eva2.pctCorrect()/100)+"\n"); 
        //System.out.println("KNN "+ i +" : "+String.valueOf(eval2.pctCorrect()/100)+"\n");   
        mediaKNN2 = mediaKNN2 + eva2.pctCorrect()/100;
       
        IBk ClassificadorKnn3 = new IBk(WEIGHT_SIMILARITY); //todos as metricas de distancia obtem a mesma media
        ClassificadorKnn3.setKNN(k);
        ClassificadorKnn3.buildClassifier(valida);
        Evaluation eva3 = new Evaluation(valida);
        eva3.evaluateModel(ClassificadorKnn3, teste);                
        gravarArq.printf(String.valueOf(eva3.pctCorrect()/100)+"\n"); 
        //System.out.println("KNN "+ i +" : "+String.valueOf(eval2.pctCorrect()/100)+"\n");   
        mediaKNN3= mediaKNN3+ eva3.pctCorrect()/100;
        
        J48 ClassificadorAD_SP = new J48();
        ClassificadorAD_SP.setUnpruned(true);
        ClassificadorAD_SP.buildClassifier(valida);
        Evaluation eval3 = new Evaluation(valida);
        eval3.evaluateModel(ClassificadorAD_SP, teste);                
        gravarArq.printf(String.valueOf(eval3.pctCorrect()/100)+"\n"); 
        //System.out.println("Árvore de Decisão"+ i +" : "+String.valueOf(eval3.pctCorrect()/100)+"\n"); 
        mediaAD_SP = mediaAD_SP + eval3.pctCorrect()/100;
        
        
        J48 ClassificadorAD_CP = new J48();
        ClassificadorAD_CP.setUnpruned(false);
        ClassificadorAD_CP.buildClassifier(valida);
        Evaluation eva22 = new Evaluation(valida);
        eva22.evaluateModel(ClassificadorAD_CP, teste);                
        gravarArq.printf(String.valueOf(eva22.pctCorrect()/100)+"\n"); 
        //System.out.println("Árvore de Decisão"+ i +" : "+String.valueOf(eval3.pctCorrect()/100)+"\n"); 
        mediaAD_CP = mediaAD_CP + eva22.pctCorrect()/100;
         breader.close();  

     }
     
     mediaMLP = mediaMLP/20;
   
     mediaSMO_RBK = mediaSMO_RBK/20;
     mediaSMO_Poly= mediaSMO_Poly/20;

     mediaKNN1 = mediaKNN1/20;
     mediaKNN2 = mediaKNN2/20;
     mediaKNN3 = mediaKNN3/20;
     
     mediaAD_SP = mediaAD_SP/20;
     mediaAD_CP = mediaAD_CP/20;
     
     System.out.println("Media MlP:" + mediaMLP);
    
     System.out.println("Media SMO RBK:" + mediaSMO_RBK);
     System.out.println("Media SMO Poly:" + mediaSMO_Poly);

     System.out.println("Media KNN none:" + mediaKNN1);
     System.out.println("Media KNN inversw:" + mediaKNN2);
     System.out.println("Media KNN simi:" + mediaKNN3);


     System.out.println("Media AD_sp:" + mediaAD_SP);
     System.out.println("Media AD_cp:" + mediaAD_CP);
    
    }
    
    public static void main(String[] args) throws IOException, Exception {
     
      //carrega os dados de treino 
     double mediaNB = 0, mediaMLP = 0, mediaSMO = 0, mediaKNN = 0, mediaAD = 0;
     double C = 0;
     Random gerador = new Random();
     int x = gerador.nextInt(20) +1;
     int k = 0;
             
     //decidindoMelhoresParametros(x);

     for(int i = 0; i< 20; i++){
        BufferedReader  breader = null;    
        breader = new BufferedReader(new FileReader("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\Banana.arff"));    
        Instances todos = new Instances (breader);
        todos.setClassIndex(todos.numAttributes()-1); //identifica que a classe está na última coluna

        //faz a separação em 50% para treino e 50% para o restante (temp)      
        StratifiedRemoveFolds fold = new StratifiedRemoveFolds();
        fold.setInputFormat( todos );
        fold.setSeed( x + i );//Defina a semente para geração de números aleatórios
        fold.setNumFolds( 2 );
        fold.setFold( 1 );
        Instances treino = Filter.useFilter( todos, fold );
        
        //depois de separado, o conjunto de treino é gravado em disco em formato .arff
        FileWriter arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaTreino.arff"); 
        PrintWriter gravarArq = new PrintWriter(arq);
        String Temp = treino.toString();
        gravarArq.printf(Temp);
        arq.close();    
        
        fold = new StratifiedRemoveFolds();
        fold.setInputFormat( todos );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 2 );        
        Instances temporario = Filter.useFilter( todos, fold );   
        
        //a segunda metade da base original é dividida em teste e validação
        fold.setInputFormat( temporario );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 1 );        
        Instances valida = Filter.useFilter(temporario, fold );
        
        //o conjunto de validação é gravado em disco em formato .arff     
        arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaValidacao.arff"); 
        gravarArq = new PrintWriter(arq);
        Temp = valida.toString();
        gravarArq.printf(Temp);
        arq.close();     
                
        fold.setInputFormat( temporario );
        fold.setSeed( x + i );
        fold.setNumFolds( 2 );
        fold.setFold( 2 );        
        Instances teste = Filter.useFilter( temporario, fold );
        
        //o conjunto de teste é gravado em disco em formato .arff     
        arq = new FileWriter("C:\\Users\\bianc\\Documents\\Banana\\AM\\src\\Base\\BananaTeste.arff"); 
        gravarArq = new PrintWriter(arq);
        Temp = teste.toString();
        gravarArq.printf(Temp);
        arq.close();            
        
        if(i==0){
            k = acharK(valida,teste);
            C = melhorC(valida,teste);
            System.out.println("K = "+k);
            System.out.println("C = "+C);
            System.out.println("Semente para geração de números aleatórios ="+ x);

           
        }
        
        //Classificadores
        
        IBk ClassificadorKnn = new IBk(WEIGHT_NONE); //todos as metricas de distancia obtem a mesma media
        ClassificadorKnn.setKNN(k);
        ClassificadorKnn.buildClassifier(treino);
        Evaluation eval2 = new Evaluation(treino);
        eval2.evaluateModel(ClassificadorKnn, teste);                
        gravarArq.printf(String.valueOf(eval2.pctCorrect()/100)+"\n"); 
        System.out.println("KNN "+ i +" : "+String.valueOf(eval2.pctCorrect()/100)+"\n");   
        mediaKNN = mediaKNN + eval2.pctCorrect()/100;
        
        J48 ClassificadorAD = new J48();
        ClassificadorAD.setUnpruned(true);
        ClassificadorAD.buildClassifier(valida);
        Evaluation eval3 = new Evaluation(valida);
        eval3.evaluateModel(ClassificadorAD, teste);                
        gravarArq.printf(String.valueOf(eval3.pctCorrect()/100)+"\n"); 
        System.out.println("Árvore de Decisão"+ i +" : "+String.valueOf(eval3.pctCorrect()/100)+"\n"); 
        mediaAD = mediaAD + eval3.pctCorrect()/100;
        
        NaiveBayes ClassificadorNB = new NaiveBayes();
        ClassificadorNB.buildClassifier(treino);
        Evaluation eval4 = new Evaluation(treino);
        eval4.evaluateModel(ClassificadorNB, teste);                
        gravarArq.printf(String.valueOf(eval4.pctCorrect()/100)+"\n"); 
        System.out.println("Naive Bayes "+ i +" : "+String.valueOf(eval4.pctCorrect()/100)+"\n");
        mediaNB = mediaNB + eval4.pctCorrect()/100;
        
        SMO ClassificadorSMOP = new SMO();
        PolyKernel pk = new PolyKernel();
        ClassificadorSMOP.setC(C);
        ClassificadorSMOP.setKernel(pk);
        ClassificadorSMOP.buildClassifier(treino);
        Evaluation eval5 = new Evaluation(treino);
        eval5.evaluateModel(ClassificadorSMOP, teste);                
        gravarArq.printf(String.valueOf(eval5.pctCorrect()/100)+"\n"); 
        System.out.println("SVM"+ i +" : "+String.valueOf(eval5.pctCorrect()/100)+"\n"); 
        mediaSMO = mediaSMO + eval5.pctCorrect()/100;
        
        MultilayerPerceptron ClassificadorMP = new MultilayerPerceptron();        
        ClassificadorMP.setLearningRate(0.1);
        ClassificadorMP.setMomentum(0.1);
        ClassificadorMP.setTrainingTime(500);
        ClassificadorMP.setHiddenLayers("5");                
        ClassificadorMP.buildClassifier(treino);
        Evaluation eval6 = new Evaluation(treino);
        eval6.evaluateModel(ClassificadorMP, teste);                
        gravarArq.printf(String.valueOf(eval6.pctCorrect()/100)+"\n"); 
        System.out.println("MLP"+ i +" : "+String.valueOf(eval6.pctCorrect()/100)+"\n");         
        mediaMLP = mediaMLP + eval6.pctCorrect()/100;
        
        
        breader.close();    

     }
     
     mediaNB = mediaNB/20;
     mediaMLP = mediaMLP/20;
     mediaSMO = mediaSMO/20;
     mediaKNN = mediaKNN/20;
     mediaAD = mediaAD/20;
     
     System.out.println("Media KNN:" + mediaKNN);
     System.out.println("Media AD:" + mediaAD);
     System.out.println("Media NB:" + mediaNB);
     System.out.println("Media SMO:" + mediaSMO);
     System.out.println("Media MlP:" + mediaMLP);
     
//
  }
    
}
