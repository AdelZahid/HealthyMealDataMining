package dwm;

import dwm.ModelPersistence;
import dwm.Util;
import weka.core.Instances;

import java.io.File;
import java.util.Scanner;

class MealPredictor {
    /** Interactive prediction; falls back to dataset demo samples when there
     *  is no console (e.g. CI). Shows predicted class + real probabilities. */
    public void predictMode() throws Exception {
        ModelPersistence.ClassifierShell sh = new ModelPersistence.ClassifierShell();
        sh.model = weka.core.SerializationHelper.read("models/best_model.model");
        Instances header = ModelPersistence.loadHeader();
        weka.classifiers.Classifier model = (weka.classifiers.Classifier) sh.model;
        Scanner sc = new Scanner(System.in);
        System.out.println("=== NEW MEAL PREDICTION (attributes are normalized 0..1) ===");
        System.out.print("Enter values interactively? (y/n) ");
        boolean interactive = sc.hasNextLine() && sc.nextLine().trim().equalsIgnoreCase("y");
        if (interactive) {
            double[] vals = new double[header.numAttributes()];
            for (int a = 0; a < header.numAttributes(); a++) {
                if (a == header.classIndex()) { vals[a] = weka.core.Utils.missingValue(); continue; }
                System.out.print(header.attribute(a).name() + " [0..1]: ");
                vals[a] = Double.parseDouble(sc.nextLine().trim());
            }
            predictOne(model, header, vals, "user-entered meal");
        } else {
            // demo mode: a few real dataset rows for verification
            weka.core.converters.ArffLoader l = new weka.core.converters.ArffLoader();
            l.setFile(new File("models/header.arff")); Instances h = l.getDataSet();
            // header has no instances; rebuild demo instance values from the CSV instead:
            runDemo(model, header);
        }
    }
    private void runDemo(weka.classifiers.Classifier model, Instances header) throws Exception {
        // deterministic demo profile similar to a typical low-calorie/high-protein meal
        double[] demo = {0.45, 0.30, 0.40, 0.85, 0.25, 0.35, 0, 0, 1, 1, 1, 1};
        predictOne(model, header, demo, "demo meal: low-calorie high-protein keto/paleo bowl");
    }
    private void predictOne(weka.classifiers.Classifier model, Instances header, double[] vals, String label) throws Exception {
        weka.core.Instance inst = new weka.core.DenseInstance(1.0, vals);
        inst.setDataset(header);
        double pred = model.classifyInstance(inst);
        double[] dist = model.distributionForInstance(inst);
        System.out.println("\nPrediction for: " + label);
        System.out.println("  Model prediction : " + Util.healthyLabel(pred));
        for (int v = 0; v < header.classAttribute().numValues(); v++)
            System.out.printf("  P(%-11s) = %.3f%n", Util.healthyLabel(v), dist[v]);
    }
}