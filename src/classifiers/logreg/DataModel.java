package classifiers.logreg;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.log4j.Logger;

import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

/**
* Train data file format:
* In the train data file, the first line is the list of unique features i.e alphabet.
* First line: feature(<COMMA>feature)+
* From second line,
* the format is: true-label(<COMMA>index of feature in alphabet<COMMA>value)+
* Test data file format:
* In the test data file, each line is a test example. Each line is of the form:
* (index of feature in alphabet<COMMA>value)+
 */

public class DataModel {
    private String delimiter = ",";
    private Logger logger = Logger.getLogger(DataModel.class);
    private String trDataFName;
    private String teDataFName;
    private Alphabet alphabet;
    private InstanceList trData;
    private InstanceList teData;
    public DataModel(String trdfn,String tedfn){
        alphabet = null;
        trData = null;
        teData = null;
        trDataFName = trdfn;
        teDataFName = tedfn;
    }
    public void readTrainDataFromFile(){
        readTrainDataFromFile(this.trDataFName);
    }
    public void readTrainDataFromFile(String trdfn){
        if(alphabet != null){
            //System.out.println("alphabet is not null");
            logger.debug("Alphabet is not null");
            return;
        }
        if(trdfn == null){
            logger.debug("train data file name is null");
            //System.out.println("train data file name is null");
            return;
        }
        try{
            BufferedReader br = new BufferedReader(new FileReader(trdfn));
            String line;
            line = br.readLine();
            if(line == null){
                //System.out.println("Alphabet is empty");
                logger.debug("Alphabet is empty");
                return;
            }
            initializeAlphabet(line);
            Pipe shellPipe = new SerialPipes(new Pipe[] { new Noop(),
                    // new PrintInput()
            });
            shellPipe.setDataAlphabet(alphabet);            
            trData = new InstanceList(shellPipe);
            while((line = br.readLine()) != null){
                int index = line.indexOf(delimiter);
                int target = Integer.parseInt(line.substring(0,index));
                trData.addThruPipe(makeInstance(target,line.substring(index + 1)));
            }
        }catch(IOException ioe){
            System.out.println(ioe.getMessage());
        }
    }
    private void initializeAlphabet(String str){
        if (alphabet != null){
            return;
        }
        alphabet = new Alphabet();
        String[] features = str.split(delimiter);
        for (String s : features){
            alphabet.lookupIndex(s);
        }
        logger.info("Alphabet size:" + alphabet.size());
        logger.info("Done loading alphabet");        
    }
    public Instance makeInstance(int target,String s) {
        try {
            String[] features = s.split(delimiter);
            int flen = features.length;
            int[] indices = new int[flen/2];
            double[] values = new double[flen/2];
            
            int id = 0;
            int k = 0;
            int previousIndex = -1;
            while (id < flen) {
                // alternate between the alphabet feature's index...
                indices[k] = Integer.parseInt(features[id]);                
                // ... and the feature value
                values[k] = Double.parseDouble(features[id + 1]);
                if (indices[k] <= previousIndex)
                    throw new RuntimeException("Vector not sorted (" + indices[k] + " <= "
                            + previousIndex + " at " + k + "): ");
                previousIndex = indices[k];
                k++;
                id += 2;
            }

            //FeatureVector fv = new FeatureVector(alphabet, indices, values,flen, flen, false, false, false);
            FeatureVector fv = new FeatureVector(alphabet, indices, values);
            //System.out.println(fv.toString());
            Instance instance = new Instance(fv,target,null, null);
            return instance;
        } catch (Exception e) {
            throw new RuntimeException("makeShellVector error: ", e);
        }
    }    
    public void readTestDataFromFile(){
        readTestDataFromFile(this.teDataFName);
    }
    public void readTestDataFromFile(String tedfn){
        if(tedfn == null){
            //System.out.println("test data file name is null");
            logger.debug("test data file name is null");
            return;
        }        
        try{
            Pipe shellPipe = new SerialPipes(new Pipe[] { new Noop(),
                    // new PrintInput(),
            });
            shellPipe.setDataAlphabet(alphabet);            
            teData = new InstanceList(shellPipe);            
            BufferedReader br = new BufferedReader(new FileReader(tedfn));
            String line;
            while((line = br.readLine()) != null){
                int index = line.indexOf(delimiter);
                int target = Integer.parseInt(line.substring(0,index));
                teData.addThruPipe(makeInstance(target,line.substring(index + 1)));
            }
        }catch(IOException ioe){
            System.out.println(ioe.getMessage());
        }
    }
    public void computeTrainingAccuracyAndWrite(Classifier classifier,String tracfn){
        assert (classifier.alphabet == trData.getDataAlphabet());
        float trainAccuracy = 0;
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(tracfn));
            bw.write("predicted<TAB>actual<TAB>features");
            for(int i = 0;i < trData.size();i++){
                Instance instance = trData.get(i);
                double pred = classifier.classify(instance);
                int predClass;
                if(pred > 0.5){
                    predClass = 1;
                }
                else{
                    predClass = 0;
                }
                int targetClass = (Integer)instance.getTarget();
                if(predClass == targetClass){
                    trainAccuracy++;
                }                
                bw.write("\n");                
                bw.write(Integer.toString(predClass));
                bw.write("\t");
                bw.write(Integer.toString(targetClass));
                bw.write("\t");
                bw.write(fvtoString((FeatureVector)instance.getData()));
            }
            trainAccuracy = (trainAccuracy * 100) / trData.size(); 
            bw.write("\n");
            bw.write("Accuracy:" + trainAccuracy);
            bw.close();
            logger.info("Done calculating training accuracy");
        }catch(IOException ioe){
            System.out.println(ioe.getMessage());
        }
    }
    public void computeTestAccuracyAndWrite(Classifier classifier,String teacfn){
        assert (classifier.alphabet == teData.getDataAlphabet());
        float testAccuracy = 0;
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter(teacfn));
            bw.write("predicted<TAB>actual<TAB>features");
            for(int i = 0;i < teData.size();i++){
                Instance instance = teData.get(i);
                double pred = classifier.classify(instance);
                int predClass;
                if(pred > 0.5){
                    predClass = 1;
                }
                else{
                    predClass = 0;
                }
                int targetClass = (Integer)instance.getTarget();
                if(predClass == targetClass){
                    testAccuracy++;
                }
                bw.write("\n");                
                bw.write(Integer.toString(predClass));
                bw.write("\t");
                bw.write(Integer.toString(targetClass));
                bw.write("\t");
                bw.write(fvtoString((FeatureVector)instance.getData()));
            }
            testAccuracy = (testAccuracy * 100) / teData.size(); 
            bw.write("\n");
            bw.write("Accuracy:" + testAccuracy);
            bw.close();
            logger.info("Done calculating test accuracy");            
        }catch(IOException ioe){
            System.out.println(ioe.getMessage());
        }        
    }
        
    public String fvtoString(FeatureVector fv){
        StringBuilder sb = new StringBuilder();
        for(int i = 0;i < fv.numLocations();i++){
            sb.append(fv.indexAtLocation(i));
            sb.append(delimiter);
            sb.append(fv.valueAtLocation(i));
            sb.append(delimiter);
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }
    
    public Alphabet getAlphabet() {
        return alphabet;
    }
    public InstanceList getTrData() {
        return trData;
    }
    public InstanceList getTeData() {
        return teData;
    }
}