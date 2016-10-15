package classifiers.logreg;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import org.apache.log4j.Logger;
import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.MatrixOps;
//import edu.cmu.ml.rtw.util.Sort;

public class LogisticRegressionClassifier extends Classifier{
    
    String name = "LogisticRegressionSGD";
    protected Logger log = Logger.getLogger(LogisticRegressionClassifier.class);
    protected Properties properties;
    private List<Integer> trainList;
    private List<Integer> heldOutList;
    private List<Integer> lamdaList;
    private List<Double> hLossList;
    public List<Double> gethLossList() {
        return hLossList;
    }

    private List<Double> trainLossList;
    public List<Double> getTrainLossList() {
        return trainLossList;
    }

    private double[] dTheta;
    private Map<Integer,Double> sTheta;
    private int[] sortedFeatureIds;
    private double bias;
    //private final int sgdBatchSize = 25;
    private Random rand;
    
    //tunable / configuration parameters
    private double l2Wt;
    private boolean isSparse;
    private double learningRate;
    private double learningRateIncreaseFactor;
    private double learningRateDecreaseFactor;
    private int MAX_ITERS;
    private byte heldOutSetSizePercentage;
    private int heldOutSetSizeValue;
    private byte learningRateSetSizePercentage;
    private int learningRateSetSizeValue;
    private int learningRateUpdateFrequency;
    private int hLossUpdateFrequency;
    
    public LogisticRegressionClassifier(Alphabet alphabet) {
        super(alphabet);
        setsTheta(null);        //initialized while reading model file
        setdTheta(null);        //initialized while reading model file
        setBias(0.0);
        setRand(new Random());
        this.sortedFeatureIds = null;
        this.trainList = null;
        this.heldOutList = null;
        this.lamdaList = null;
        hLossList = new ArrayList<Double>();
        trainLossList = new ArrayList<Double>();
        readPropertiesFile();
    }
/*    
    public LogisticRegressionClassifier(Alphabet alphabet, boolean isSparse) {
        this(alphabet);
        this.isSparse = isSparse;       //let the default be dense
    }
*/  
    public void readPropertiesFile(){
        Properties properties = new Properties();
        try {
            properties.load(LogisticRegressionClassifier.class.getResourceAsStream("LogisticRegressionClassifier.properties"));
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        this.l2Wt = Double.parseDouble(properties.getProperty("l2Wt"));
        log.info("l2Wt:" + this.l2Wt);
        this.isSparse = Boolean.parseBoolean(properties.getProperty("isSparse"));
        log.info("isSparse:" + this.isSparse);        
        this.learningRate = Double.parseDouble(properties.getProperty("learningRate"));
        log.info("learningRate:" + this.learningRate);
        this.learningRateDecreaseFactor = Double.parseDouble(properties.getProperty("learningRateDecreaseFactor"));
        this.learningRateIncreaseFactor = Double.parseDouble(properties.getProperty("learningRateIncreaseFactor"));
        this.MAX_ITERS = Integer.parseInt(properties.getProperty("MAX_ITERS"));
        log.info("MAX_ITERS:" + this.MAX_ITERS);
        this.heldOutSetSizePercentage = Byte.parseByte(properties.getProperty("heldOutSetSizePercentage"));
        this.heldOutSetSizeValue = Integer.parseInt(properties.getProperty("heldOutSetSizeValue"));
        this.learningRateSetSizePercentage = Byte.parseByte(properties.getProperty("learningRateSetSizePercentage"));
        this.learningRateSetSizeValue = Integer.parseInt(properties.getProperty("learningRateSetSizeValue"));
        this.learningRateUpdateFrequency = Integer.parseInt(properties.getProperty("learningRateUpdateFrequency"));
        this.hLossUpdateFrequency = Integer.parseInt(properties.getProperty("hLossUpdateFrequency"));
    }
    
    @Override
    public double classify(Instance instance) {
        try{
            double sum = bias;
            final FeatureVector fv = (FeatureVector) instance.getData();
            int i = 0;
            // If we're sparse...
            if (isSparse) {
                while (i < fv.numLocations()) {
                    int featurei = fv.indexAtLocation(i);
                    if(sTheta.containsKey(featurei)){
                        sum += fv.valueAtLocation(i) * sTheta.get(featurei); 
                    }
                    i++;
                }
            }
            // If we're dense...
            else {
                while(i < fv.numLocations()){
                    sum += fv.valueAtLocation(i) * dTheta[fv.indexAtLocation(i)];
                    i++;
                } 
            }
            return sigmoid(sum);
        }catch(Exception e){
            throw new RuntimeException("classify(" + instance.getData() + ")", e);
        }
    }

    @Override
    public void train(InstanceList ilist) {
        splitData(ilist.size());
        train(ilist,MAX_ITERS);
    }
    
    public void train(InstanceList ilist,int numItrs){
        log.info("Training classifier for " + numItrs + " iterations. . .");
        assert(this.alphabet == ilist.getDataAlphabet());
        if(isSparse){
            if(getsTheta() == null){
                setsTheta(new HashMap<Integer,Double>());                
            }
        }
        else{
            if(getdTheta() == null){
                setdTheta(new double[this.alphabet.size()]);
                MatrixOps.setAll(getdTheta(), 0.0);
            }            
        }
        int k = 0;
        double previousLoss = Double.POSITIVE_INFINITY;
        double currentLoss = Double.POSITIVE_INFINITY;
        double previousHLoss = Double.POSITIVE_INFINITY;
        double currentHLoss = Double.POSITIVE_INFINITY;
        
        int itr = 0;
        int terminationCount1 = 0;
        int terminationCount2 = 0;
        //create a list to be shuffled every iteration
        Map<Integer,Integer> regularizeCounter = new HashMap<Integer,Integer>();
        //while(currentLoss > 0.45 && itr < numItrs){
        while(itr < numItrs && terminationCount1 < 5 && terminationCount2 < 5){
            Collections.shuffle(trainList);
            for(int i = 0;i < trainList.size();i++){
                Instance instance = ilist.get(trainList.get(i));
                k = k + 1;
                //double instanceWeight = ilist.getInstanceWeight(instance);
                double real = (Integer) instance.getTarget();
                double pred = classify(instance);
                FeatureVector fv = (FeatureVector) instance.getData();
                int ii = 0;
                //update bias
                this.bias = this.bias + this.learningRate * ((real - pred) - (2 * this.l2Wt * this.bias));
                while(ii < fv.numLocations()){
                    int j = fv.indexAtLocation(ii);
                    if(!regularizeCounter.containsKey(j)){
                        regularizeCounter.put(j,0);
                    }
                    if(isSparse){
                        if(!sTheta.containsKey(j)){
                            sTheta.put(j, 0.0);
                        }
                        double temp = sTheta.get(j);
                        temp = temp * Math.pow((1 - 2 * this.learningRate * this.l2Wt), k - regularizeCounter.get(j));
                        temp = temp + this.learningRate * (real - pred) * fv.valueAtLocation(ii);
                        sTheta.put(j,temp);
                    }
                    else{
                        dTheta[j] = dTheta[j] * Math.pow((1 - 2 * this.learningRate * this.l2Wt), k - regularizeCounter.get(j));
                        dTheta[j] = dTheta[j] + this.learningRate * (real - pred) * fv.valueAtLocation(ii);                    
                    }
                    regularizeCounter.put(j,k);
                    ii++;
                }
                if(k % this.learningRateUpdateFrequency == 0){
                    previousLoss = currentLoss;
                    currentLoss = computeLoss(ilist,lamdaList);
                    trainLossList.add(currentLoss);
                    log.info("After " + k + " gradient descent steps," + " current loss:" + currentLoss + ", previous loss:" + previousLoss);
//                  learningRate = learningRate / 1.2;                    
                    if(currentLoss < previousLoss){
                        //additively increase the learning rate
                        learningRate = learningRate * this.learningRateIncreaseFactor;
                    }
                    else{
                        //multiplicatively decrease the learning rate
                        learningRate = learningRate / this.learningRateDecreaseFactor;
                    }
                    log.info("After " + k + " gradient descent steps, learning rate:" + learningRate);                    
                }
                if(k % this.hLossUpdateFrequency == 0){
                    previousHLoss = currentHLoss;
                    currentHLoss = computeLoss(ilist,heldOutList);
                    if(Math.abs(currentHLoss - previousHLoss) < 1e-3){
                        terminationCount1++;
                    }
                    else{
                        terminationCount1 = 0;
                    }
                    if(currentHLoss - previousHLoss > 1e-3){
                        terminationCount2++;
                    }
                    else{
                        terminationCount2 = 0;
                    }                    
                    hLossList.add(currentHLoss);
                    log.info("After " + k + " gradient descent steps," + " current held out loss:" + currentHLoss + ", previous held out loss:" + previousHLoss);
                }
            }
            itr++;
        }
        //update theta for last time
        if(isSparse){
            int exponent;
            Set<Integer> indexes = sTheta.keySet();
            for(int index : indexes){
                if(regularizeCounter.containsKey(index)){
                    exponent = k - regularizeCounter.get(index);
                }
                else{
                    exponent = k;
                }
                double temp = sTheta.get(index);
                temp = temp * Math.pow((1 - 2 * this.learningRate * this.l2Wt), exponent);
                sTheta.put(index,temp);
            }
            //if isSparse = true then sTheta will have changed so make sortedFeatureIds = null            
            this.sortedFeatureIds = null;
        }
        else{
            int exponent;
            for(int c = 0;c < dTheta.length;c++){
                if(regularizeCounter.containsKey(c)){
                    exponent = k - regularizeCounter.get(c);
                }
                else{
                    exponent = k;
                }
                dTheta[c] = dTheta[c] * Math.pow((1 - 2 * this.learningRate * this.l2Wt), exponent);                
            }
        }
        writeLosses();
    }
/*    
    public int getRandomNumber(int min,int max){
        return rand.nextInt((max - min) + 1) + min;
    }
 */
    private double sigmoid(final double sum) {
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    public double computeLoss(InstanceList ilist,List<Integer> list){
        //got to implement this method
        double loss = 0.0;
        for(int i = 0;i < list.size();i++){
            Instance instance = ilist.get(list.get(i));
            //k = k + 1;
            //double instanceWeight = ilist.getInstanceWeight(instance);
            double real = (Integer) instance.getTarget();
            double pred = classify(instance);
//            assert(!(real == 1.0 && pred == 0.0)) : "Bad prediction" + "(real: " + real + ",pred: " + pred + ")";
//            assert(!(real == 0.0 && pred == 1.0)) : "Bad prediction" + "(real: " + real + ",pred: " + pred + ")";            
            //loss -= (instanceWeight * Math.log((real > 0.5 ? pred : 1 - pred)));
            loss -= Math.log((real > 0.5 ? pred : 1 - pred));            
        }
        loss = loss / list.size();
        double thetaSqr = Math.pow(bias, 2.0);
        if(isSparse){
            Set<Integer> indexes = sTheta.keySet();
            for(int index : indexes){
                thetaSqr += Math.pow(sTheta.get(index), 2.0);
            }
        }
        else{
            for(int i = 0;i < dTheta.length;i++){
                thetaSqr += Math.pow(dTheta[i], 2.0);
            }
        }
        loss += this.l2Wt * thetaSqr;
        return loss;
    }
    
    public void splitData(int iListSize){
        if(this.trainList != null && this.heldOutList != null && this.lamdaList != null){
            return;
        }
        trainList = new ArrayList<Integer>(iListSize);
        for(int i = 0;i < iListSize;i++){
            trainList.add(i);
        }
        Collections.shuffle(trainList);
        //initialize lamdaSet & heldOutSet
        heldOutList = new ArrayList<Integer>();
        if(heldOutSetSizePercentage != -1){
            int hoSize = (heldOutSetSizePercentage * iListSize) / 100;
            int temp1 = trainList.size();            
            for(int i = 0;i < hoSize;i++){
                int rnum = (int)(temp1 * rand.nextDouble());
                int temp2 = trainList.get(rnum);
                trainList.set(rnum, trainList.get(trainList.size() - 1));
                trainList.remove(trainList.size() - 1);
                heldOutList.add(temp2);
                temp1--;                
            }
        }
        else if(heldOutSetSizeValue != -1){
            if(heldOutSetSizeValue > trainList.size()){
                log.error("heldOutSetSizeValue exceeds train data size" + "(" + trainList.size() +")");
            }
            int temp1 = trainList.size();                        
            for(int i = 0;i < heldOutSetSizeValue;i++){
                int rnum = (int)(temp1 * rand.nextDouble());
                int temp2 = trainList.get(rnum);
                trainList.set(rnum, trainList.get(trainList.size() - 1));
                trainList.remove(trainList.size() - 1);
                heldOutList.add(temp2);
                temp1--;                                
            }            
        }
        else{
            log.error("Both heldOutSetSizePercentage & heldOutSetSizeValue can not be equal to -1 at the same time");            
        }
        assert(trainList.size() == iListSize - heldOutList.size());
        Collections.shuffle(trainList);
        lamdaList = new ArrayList<Integer>();
        if(learningRateSetSizePercentage != -1){
            int lrsSize = (learningRateSetSizePercentage * iListSize) / 100;
            int temp1 = trainList.size();
            for(int i = 0;i < lrsSize;i++){
                int rnum = (int)(temp1 * rand.nextDouble());
                int temp2 = trainList.get(rnum);
                trainList.set(rnum, trainList.get(trainList.size() - 1));
                trainList.set(trainList.size() - 1, temp2);
                lamdaList.add(temp2);
                temp1--;
            }            
        }
        else if(learningRateSetSizeValue != -1){
            if(learningRateSetSizeValue > trainList.size()){
                log.error("learningRateSetSizeValue exceeds train data size" + "(" + trainList.size() +")");
            }
            int temp1 = trainList.size();
            
            for(int i = 0;i < learningRateSetSizeValue;i++){
                int rnum = (int)(temp1 * rand.nextDouble());
                int temp2 = trainList.get(rnum);
                trainList.set(rnum, trainList.get(trainList.size() - 1));
                trainList.set(trainList.size() - 1, temp2);
                lamdaList.add(temp2);
                temp1--;                
            }                        
        }
        else{
            log.error("Both learningRateSetSizePercentage & learningRateSetSizeValue can not be equal to -1 at the same time");
        }
        assert(trainList.size() == iListSize - heldOutList.size());
        log.info("held out set size:" + heldOutList.size());
        log.info("lamda set size:" + lamdaList.size());
    }
    
    public boolean isSparse(){
        return isSparse;
    }
    
    public void setSparse(){
        if(getsTheta() != null){
            return;
        }
        isSparse = true;
        setsTheta(new HashMap<Integer,Double>());            
        for(int i = 0;i < this.dTheta.length;i++){
            if (dTheta[i] < -0.00000001 || dTheta[i] > 0.00000001){
                if(!sTheta.containsKey(i)){
                    sTheta.put(i, dTheta[i]);
                }
            }
        }
        setdTheta(null);
    }
    
    public void setDense(){
        if(getdTheta() != null){
            return;
        }
        isSparse = false;
        setdTheta(new double[this.alphabet.size()]);
        MatrixOps.setAll(getdTheta(), 0.0);
        Set<Integer> ids = sTheta.keySet();
        for(int id : ids){
            dTheta[id] = sTheta.get(id);
        }
        setsTheta(null);
    }

    @Override
    public String getModelSummary() {
        // TODO Auto-generated method stub
        return getModelSummary(40);
    }

    public String getModelSummary(int numToReturn) {
        int numFeatures;
        int[] sortedFeatures;
        double[] sortedParams;
        if(isSparse){
            numFeatures = sTheta.size();
            sortedFeatures = new int[numFeatures];
            sortedParams = new double[numFeatures];
            int count = 0;
            for (Map.Entry<Integer, Double> entry : sTheta.entrySet()){
                sortedFeatures[count] = entry.getKey();
                sortedParams[count] = entry.getValue();
                count++;
            }
        }else{
            numFeatures = 0;
            for (int i = 0; i < dTheta.length; i++) {
                if (dTheta[i] < -0.00000001 || dTheta[i] > 0.00000001) numFeatures++;
            }
            sortedFeatures = new int[numFeatures];
            sortedParams = new double[numFeatures];
            int j = 0;
            for (int i = 0; i < dTheta.length; i++) {
                if (dTheta[i] < -0.00000001 || dTheta[i] > 0.00000001) {
                    sortedFeatures[j] = i;
                    sortedParams[j] = dTheta[i];
                    j++;
                }
            }
        }
            
        //Sort.heapify(sortedParams, sortedFeatures);
        //Sort.heapsort(sortedParams, sortedFeatures);

        // Use up to 1/3 of our allotment (rounding down) on the most heavily-weighted negative
        // features.
        StringBuffer sb = new StringBuffer();
        int numUsed = 0;
        int numNegativesToUse = numToReturn / 3;
        for (int rank = 0; rank < numFeatures && numUsed < numNegativesToUse; rank++) {
            int feature = sortedFeatures[rank];
            double value = sortedParams[rank];
            if (value >= 0.0) break;
            if (value > -0.000001) continue;
            sb.append(alphabet.lookupObject(feature) + String.format("\t%.5f\t", value));
            numUsed++;
        }

        // The remainder go to the most heavily-weighted positive features.  If we don't use up our
        // entire allotment, that's OK; this is just a summary and we're making a decent effort at
        // it.
        for (int rank = numFeatures - (numToReturn - numUsed) - 1; rank < numFeatures && numUsed < numToReturn; rank++) {
            if (rank < 0) {
                if (numFeatures == 0) break;
                rank = 0;
            }
            int feature = sortedFeatures[rank];
            double value = sortedParams[rank];
            if (value < 0.000001) continue;
            sb.append(alphabet.lookupObject(feature) + String.format("\t%.5f\t", value));
            numUsed++;
        }

        return sb.toString().trim();
    }
    
    @Override
    public String getProvenance(Instance instance) {
        return getProvenance(instance, 10);
    }
    
    public String getProvenance(Instance instance, int numToReturn) {

        // This might be too slow and ugly, but we can come back and make it more lean and mean as
        // needed.  One thing we could do that might be really clever is to just keep running lists
        // of our most extreme features rather than building up the whole dot product, guessing at
        // its ultimate length, sorting, etc.
        //
        // What we'll do instead is just make a products array that is parallel to our existing
        // features array, and accept that some or all of it might wind up zero because we don't
        // emit any products that are too close to zero anyway.

        int numFeatures;
        double[] products;
        int[] sortedFeatures;

        if (isSparse) {
            numFeatures = sTheta.size();
            products = new double[numFeatures];
            sortFeatureIds();
            sortedFeatures = this.sortedFeatureIds;
            FeatureVector fv = (FeatureVector) instance.getData();
            int i = 0,j = 0;
            while (i < fv.numLocations() && j < numFeatures) {
                int featurei = fv.indexAtLocation(i);
                int featurej = sortedFeatures[j];
                if (featurei == featurej) {
                    products[j] = fv.valueAtLocation(i) * sTheta.get(featurej);
                    i++;
                    j++;
                } else if (featurei < featurej) {
                    i++;
                } else {
                    j++;
                }
            }
            //System.arraycopy(sFeatures, 0, sortedFeatures, 0, numFeatures);
        } else {
            numFeatures = dTheta.length;
            products = new double[numFeatures];
            sortedFeatures = new int[numFeatures];
            FeatureVector fv = (FeatureVector) instance.getData();
            for (int i = 0; i < fv.numLocations(); i++) {
                int di = fv.indexAtLocation(i);
                products[i] = fv.valueAtLocation(i) * dTheta[di];
                sortedFeatures[i] = di;
            }
        }

        // Make a copy of our features to sort with and sort
        //Sort.heapify(products, sortedFeatures);
        //Sort.heapsort(products, sortedFeatures);

        // Use up to 1/3 of our allotment (rounding down) on the most heavily-weighted negative
        // features.
        StringBuffer sb = new StringBuffer();
        int numUsed = 0;
        int numNegativesToUse = numToReturn / 3;
        for (int rank = 0; rank < numFeatures && numUsed < numNegativesToUse; rank++) {
            int feature = sortedFeatures[rank];
            double value = products[rank];
            if (value >= 0.0) break;
            if (value > -0.000001) continue;
            sb.append(alphabet.lookupObject(feature) + String.format("\t%.5f\t", value));
            numUsed++;
        }

        // The remainder go to the most heavily-weighted positive features.  If we don't use up our
        // entire allotment, that's OK; this is just a summary and we're making a decent effort at
        // it.
        for (int rank = numFeatures - (numToReturn - numUsed) - 1; rank < numFeatures && numUsed < numToReturn; rank++) {
            int feature = sortedFeatures[rank];
            double value = products[rank];
            if (value < 0.0) break;
            if (value < 0.000001) continue;
            sb.append(alphabet.lookupObject(feature) + String.format("\t%.5f\t", value));
            numUsed++;
        }

        return sb.toString().trim();
    }

    @Override
    public void readModelFromFile(File modelFile, int[] alphabetMap) {
        if(!modelFile.exists()){
            log.debug("Model file does not exist");
            return;
        }
        try{
            BufferedReader br = new BufferedReader(new FileReader(modelFile));
            String line = br.readLine();
            if (line == null) {
                log.debug(modelFile + " is empty.  Setting empty model vector with a bias of NaN");
                setBias(Double.NaN);
                return;
            }
            int numFeatures = Integer.parseInt(line);
            int numFeaturesFound = 0;
            line = br.readLine();
            if(line == null){
                log.warn(modelFile + " ends before bias.  Setting a bias of NaN");
                setBias(Double.NaN);
                return;
            }
            else{
                setBias(Double.parseDouble(line));
            }
            if(isSparse){
                setsTheta(new HashMap<Integer,Double>());
                while ((line = br.readLine()) != null) {
                    numFeaturesFound++;
                    if (numFeaturesFound > numFeatures){
                        throw new RuntimeException(modelFile + " has more than the " + numFeatures
                                + " features it claims to have");                    
                    }
                    int pos = line.indexOf("\t");
                    int feature = Integer.parseInt(line.substring(0, pos));
                    double weight = Double.parseDouble(line.substring(pos+1));
//                    if (alphabetMap != null) {
//                        feature = alphabetMap[feature];
//                        if (feature < 0) continue;
//                    }
                    if(!sTheta.containsKey(feature)){
                        sTheta.put(feature, weight);
                    }
                }                                
            }
            else{
                setdTheta(new double[this.alphabet.size()]);
                MatrixOps.setAll(getdTheta(), 0.0);
                while ((line = br.readLine()) != null) {
                    numFeaturesFound++;
                    if (numFeaturesFound > numFeatures){
                        throw new RuntimeException(modelFile + " has more than the " + numFeatures
                                + " features it claims to have");                    
                    }
                    int pos = line.indexOf("\t");
                    int feature = Integer.parseInt(line.substring(0, pos));
                    double weight = Double.parseDouble(line.substring(pos+1));
//                    if (alphabetMap != null) {
//                        feature = alphabetMap[feature];
//                        if (feature < 0) continue;
//                    }
                    dTheta[feature] = weight;
                }                
            }
            br.close();
            if (numFeaturesFound < numFeatures){
                throw new RuntimeException(modelFile + " has fewer than the " + numFeatures
                        + " features it claims to have");                
            }
        }catch(IOException ioe){
            System.out.println("While reading model file " + modelFile);
            System.out.println(ioe.getMessage());
        }
    }
    @Override
    public void writeModelToFile(File modelFile) {
        try {
            FileWriter fw = new FileWriter(modelFile);
            if (isSparse) {
                // First line is number of features
                fw.write(sTheta.size() + "\n");

                // Next line is bias
                fw.write(String.format("%.8f\n", this.bias));

                for (int fid : sTheta.keySet())
                    fw.write(String.format("%d\t%.8f\n", fid, sTheta.get(fid)));
            } else {
                // See comments above.

                int numNonZeros = 0;
                for (int i = 0; i < dTheta.length - 1; i++){
                    if (dTheta[i] < -0.00000001 || dTheta[i] > 0.00000001) numNonZeros++;
                }

                fw.write(numNonZeros + "\n");
                fw.write(String.format("%.8f\n", this.bias));

                for (int i = 0; i < dTheta.length - 1; i++) {
                    if (dTheta[i] < -0.00000001 || dTheta[i] > 0.00000001) {
                        fw.write(String.format("%d\t%.8f\n", i, dTheta[i]));
                    }
                }
            }
            fw.close();
        }catch (IOException e) {
            throw new RuntimeException("writeModelToFile(" + modelFile + ")", e);
        }
    }
    
    public void writeLosses(){
        try{
            BufferedWriter bw = new BufferedWriter(new FileWriter("held_out_loss.txt"));
            for(int i = 0;i < hLossList.size();i++){
                bw.write(Double.toString(hLossList.get(i)));
                bw.write("\n");
            }
            bw.close();
            bw = new BufferedWriter(new FileWriter("train_loss.txt"));
            for(int i = 0;i < trainLossList.size();i++){
                bw.write(Double.toString(trainLossList.get(i)));
                bw.write("\n");                
            }
            bw.close();
        }catch(IOException ioe){
            System.out.println(ioe.getMessage());
        }
    }        
    
    public void sortFeatureIds(){
        if(isSparse){
            if(this.sortedFeatureIds != null){
                return;
            }
            else{
                Set<Integer> fids = sTheta.keySet();
                this.sortedFeatureIds = new int[fids.size()];
                int i = 0;
                for(int id : fids){
                    this.sortedFeatureIds[i] = id;
                    i++;
                }
                Arrays.sort(this.sortedFeatureIds);                
            }
        }
    }
        
    public double getL2Wt() {
        return l2Wt;
    }

    public void setL2Wt(double l2Wt) {
        this.l2Wt = l2Wt;
    }

    public double[] getdTheta() {
        return dTheta;
    }

    public void setdTheta(double[] theta) {
        this.dTheta = theta;
    }
    

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public Random getRand() {
        return rand;
    }

    public void setRand(Random rand) {
        this.rand = rand;
    }

    public Map<Integer,Double> getsTheta() {
        return sTheta;
    }

    public void setsTheta(Map<Integer,Double> sTheta) {
        this.sTheta = sTheta;
    }
}