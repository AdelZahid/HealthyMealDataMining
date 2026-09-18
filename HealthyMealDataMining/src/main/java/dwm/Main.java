package dwm;
import weka.core.Instances;
import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Util.ensureDirs();
        String mode = args.length > 0 ? args[0] : "full";
        if (mode.equals("predict")) { new MealPredictor().predictMode(); return; }
        DataLoader loader = new DataLoader();
        Instances raw = loader.loadCSV(Util.CSV_PATH);
        DataInspector.Report rep = new DataInspector().inspect(raw);
        System.out.println(rep.summary);
        Util.writeFile("results/dataset_summary.txt", rep.summary);

        DataPreprocessor pp = new DataPreprocessor();
        Instances master = pp.master(raw);
        pp.saveARFF(master, Util.ARFF_PATH);            // requirement: CSV -> ARFF
        Util.writeFile("results/preprocessing/preprocessing_summary.txt",
                "Filters applied (WEKA 3.8.6, seed=1):\n" +
                        "1. ReplaceMissingValues (0 missing found)\n" +
                        "2. StringToNominal on meal_name\n" +
                        "3. NumericToNominal on is_healthy -> nominal {0=Not Healthy, 1=Healthy}\n" +
                        "4. Normalize applied INSIDE FilteredClassifier per CV fold and on clustering view (leakage-safe)\n" +
                        "5. SMOTE: applied because imbalance ratio is " + Util.F2.format(rep.imbalanceRatio) +
                        ":1 (Healthy = " + rep.classCounts[1] + "/" + rep.rows + "). SMOTE runs only on training folds via FilteredClassifier.\n" +
                        "6. meal_name excluded from model features (identifier, not a nutritional attribute; kept in ARFF/EDA/clusters).\n" +
                        "7. Report mentions sugar & fiber - NOT PRESENT in dataset; not fabricated; analysis uses available nutrition columns.\n");

        new EDAAnalyzer().run(master, rep, "meal_name");

        Instances clsData = pp.classificationView(master);
        ClassificationEngine ce = new ClassificationEngine();
        List<ClassificationEngine.ModelResult> results = ce.runAll(clsData);
        ce.writeResults(results);
        ClassificationEngine.ModelResult best = results.stream()
                .max(Comparator.comparingDouble(r -> Double.isNaN(r.f1) ? -1 : r.f1)).orElseThrow();
        System.out.printf("%nBest model: %s%s  acc=%s F1=%s AUC=%s%n",
                best.name, best.smote?" +SMOTE":"", Util.fmt(best.acc), Util.fmt(best.f1), Util.fmt(best.auc));

        ClusteringEngine cl = new ClusteringEngine();
        cl.run(pp.clusteringView(master), master, rep);

        AssociationRuleMiner arm = new AssociationRuleMiner();
        arm.run(pp.aprioriView(master));

        new PatternAnalyzer().run(results, best.importance,
                readLines("results/association_rules/important_rules.csv"),
                readFile("results/clustering/cluster_analysis.txt"),
                100.0 * rep.classCounts[1] / rep.rows);

        ModelPersistence.ClassifierShell shell = new ModelPersistence.ClassifierShell();
        shell.model = best.fullModel; shell.header = clsData; shell.name = best.name; shell.smote = best.smote;
        new ModelPersistence().save(shell);
        Util.writeFile("results/final_analysis/final_report.txt", buildFinalReport(rep, results, best));
        System.out.println("Pipeline complete. See results/ and models/best_model.model");
        if (mode.equals("predict-after")) new MealPredictor().predictMode();
    }
    private static String readFile(String p) throws Exception {
        return new String(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(p)));
    }
    private static List<String> readLines(String p) throws Exception {
        return java.nio.file.Files.readAllLines(java.nio.file.Paths.get(p));
    }
    private static String buildFinalReport(DataInspector.Report rep,
                                           List<ClassificationEngine.ModelResult> rs, ClassificationEngine.ModelResult best) throws Exception {
        StringBuilder sb = new StringBuilder("FINAL INTEGRATED REPORT\n=======================\n\n");
        sb.append("A. DATASET\n").append(readFile("results/dataset_summary.txt")).append('\n');
        sb.append("B. PREPROCESSING\n").append(readFile("results/preprocessing/preprocessing_summary.txt")).append('\n');
        sb.append("C. EDA\n").append(readFile("results/eda/nutritional_analysis.txt")).append('\n');
        sb.append("D. CLASSIFICATION (stratified 10-fold CV, seed=1)\n").append(readFile("results/classification/model_comparison.csv")).append('\n');
        sb.append(readFile("results/classification/confusion_matrices.txt")).append('\n');
        sb.append("E. FEATURE ANALYSIS\n").append(readFile("results/classification/importance_"
                + best.name.toLowerCase() + (best.smote?"_smote":"_baseline") + ".txt"));
        if (best.name.equals("J48")) sb.append('\n').append(readFile("results/classification/j48_tree.txt"));
        sb.append("\nF. CLUSTERING\n").append(readFile("results/clustering/cluster_analysis.txt")).append('\n');
        sb.append("G. ASSOCIATION RULES\n").append(readFile("results/association_rules/rule_analysis.txt")).append('\n');
        sb.append("H. INTEGRATED PATTERNS\n").append(readFile("results/final_analysis/nutritional_patterns.txt")).append('\n');
        sb.append("I. FINAL MODEL\n  Selected: ").append(best.name).append(best.smote?" (SMOTE)":" (baseline)")
                .append(String.format("%n  Criterion: highest Healthy-class F1 under stratified 10-fold CV (F1=%s, AUC=%s)%n", Util.fmt(best.f1), Util.fmt(best.auc)))
                .append("  Serialized: models/best_model.model (+ header.arff + preprocessing.properties)\n");
        sb.append("\nJ. PREDICTION DEMONSTRATION\n  Run: java -cp out:lib/weka.jar dwm.Main predict\n");
        return sb.toString();
    }
}