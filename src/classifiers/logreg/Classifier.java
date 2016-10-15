package classifiers.logreg;

import java.io.File;

import cc.mallet.types.Alphabet;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

public abstract class Classifier {
    protected String name = "generic";
    // feature alphabet (string <--> ID mapping)
    protected Alphabet alphabet;

    public Classifier(Alphabet alphabet) {
        this.alphabet = alphabet;
    }

    // take an instance and return a posterior class estimate
    public abstract double classify(Instance instance);

    // take an InstanceList and train the classifier
    public abstract void train(InstanceList ilist);

    public abstract String getModelSummary();
    public abstract String getProvenance(Instance instance);
    
    // model IO stuff ...
    public void readModelFromFile(String modelFileName) {
        readModelFromFile(new File(modelFileName));
    }

    public void writeModelToFile(String modelFileName) {
        writeModelToFile(new File(modelFileName));
    }

    public void readModelFromFile(File modelFile) {
        readModelFromFile(modelFile, null);
    }

    public abstract void readModelFromFile(File modelFile, int[] alphabetMap);
    public abstract void writeModelToFile(File modelFile);
}