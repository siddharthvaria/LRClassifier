package classifiers.logreg;


import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Logger;

public class ClassifierTest {
    
    private static Option trainFName;
    private static Option testFName;
    private static Option tracFName;
    private static Option teacFName;
    private static Option opFName;
    private static Option npFName;
    private static Option help;
    protected static Logger log = Logger.getLogger(ClassifierTest.class);
    /**
     * @param args
     */
    @SuppressWarnings("static-access")
    public static void main(String[] args) {
        Options options = new Options();
        trainFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path of train data file").create("tr");
        testFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path of test data file").create("te");
        tracFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path of train accuracy output file").create("trac");
        teacFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path of test accuracy output file").create("teac");
        opFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path of parameters file").create("op");
        npFName = OptionBuilder.withArgName("file").hasArg().withDescription("full path where new parameters should be written").create("np");        
        help = new Option("help",false,"print help message");
        options.addOption(trainFName);
        options.addOption(testFName);
        options.addOption(tracFName);
        options.addOption(teacFName);
        options.addOption(opFName);
        options.addOption(npFName);
        options.addOption(help);
        HelpFormatter formatter = new HelpFormatter();        
        CommandLineParser parser = new BasicParser();
        CommandLine cl = null;
        try{
            cl = parser.parse(options, args);
        }catch(ParseException pe){
            System.out.println(pe.getMessage());
        }
        if(cl.hasOption("help")){
            formatter.printHelp(ClassifierTest.class.getName(), options,true);
            System.exit(0);            
        }
        BasicConfigurator.configure();
        DataModel dm = new DataModel(cl.getOptionValue("tr"),cl.getOptionValue("te"));
        dm.readTrainDataFromFile();
        dm.readTestDataFromFile();
        Classifier classifier = new LogisticRegressionClassifier(dm.getAlphabet());
        if(cl.hasOption("op")){
            classifier.readModelFromFile(cl.getOptionValue("op"));
            log.info("Initialized parameters from specified file");
        }
        classifier.train(dm.getTrData());
        if(cl.hasOption("np")){
            classifier.writeModelToFile(cl.getOptionValue("np"));
            log.info("Parameters written to specified file");
        }        
        dm.computeTrainingAccuracyAndWrite(classifier,cl.getOptionValue("trac"));
        dm.computeTestAccuracyAndWrite(classifier, cl.getOptionValue("teac"));
    }
}