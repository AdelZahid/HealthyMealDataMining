package dwm;
import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.util.*;

public class ModelPersistence
{
    public void save(ClassifierShell s) throws Exception {
        weka.core.SerializationHelper.write("models/best_model.model", s.model);
        ArffSaver sv = new ArffSaver(); sv.setInstances(s.header); sv.setFile(new File("models/header.arff")); sv.writeBatch();
        java.util.Properties p = new java.util.Properties();
        p.setProperty("seed", String.valueOf(Util.SEED));
        p.setProperty("weka.version", "3.8.6"); p.setProperty("java.version", System.getProperty("java.version"));
        p.setProperty("model", s.name); p.setProperty("smote", String.valueOf(s.smote));
        p.setProperty("note", "Model is a FilteredClassifier(Normalize[+SMOTE] -> base); input instance must match header.arff (meal_name removed). Class: 0=Not Healthy, 1=Healthy.");
        p.store(new java.io.FileOutputStream("models/preprocessing.properties"), "HealthyMealDataMining reproducibility config");
    }
    public static class ClassifierShell { public Object model; public Instances header; public String name; public boolean smote; }

    public static Instances loadHeader() throws Exception {
        ArffLoader l = new ArffLoader(); l.setFile(new File("models/header.arff"));
        Instances h = l.getDataSet(); h.setClassIndex(h.numAttributes()-1); return h;
    }
}

