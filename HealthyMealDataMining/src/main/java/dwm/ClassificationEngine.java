package dwm;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.*;

public class ClassificationEngine {

    public static class ModelResult {
        public String name;
        public boolean smote;
        public double acc, prec, rec, f1, auc;
        public int[][] cm;
        public Classifier fullModel;
        public Map<String, Double> importance = new LinkedHashMap<>();
        public String detail;
    }

    public interface Factory {
        Classifier build();
    }

    public List<ModelResult> runAll(Instances data) throws Exception {
        List<ModelResult> results = new ArrayList<>();

        Map<String, Factory> bases = new LinkedHashMap<>();
        bases.put("RandomForest", () -> {
            RandomForest r = new RandomForest();
            r.setSeed(Util.SEED);
            return r;
        });
        bases.put("J48", J48::new);
        bases.put("SMO", () -> {
            SMO s = new SMO();
            s.setBuildCalibrationModels(true);
            return s;
        });

        for (Map.Entry<String, Factory> e : bases.entrySet()) {
            results.add(eval(e.getKey(), false,
                    wrap(e.getValue(), false), data, e.getValue()));

            results.add(eval(e.getKey(), true,
                    wrap(e.getValue(), true), data, e.getValue()));
        }

        return results;
    }

    /**
     * Creates the classifier used during evaluation.
     *
     * Baseline: Normalize -> classifier.
     * SMOTE: SMOTE -> Normalize -> classifier.
     *
     * SMOTE is kept inside FilteredClassifier so it is applied separately
     * to each training fold during cross-validation.
     */
    private Classifier wrap(Factory factory, boolean useSmote) {
        FilteredClassifier fc = new FilteredClassifier();

        if (useSmote) {
            // SMOTE is a supervised Weka filter and uses the class attribute.
            // The class must already be nominal in the preprocessed data.
            weka.filters.supervised.instance.SMOTE smote =
                    new weka.filters.supervised.instance.SMOTE();

            smote.setRandomSeed(Util.SEED);
            // Apply SMOTE first, then normalize the resulting training fold.
            Normalize normalize = new Normalize();

            Filter[] filters = new Filter[]{smote, normalize};

            // FilteredClassifier accepts one Filter, so use a MultiFilter
            // for the two-step SMOTE -> Normalize pipeline.
            weka.filters.MultiFilter multi = new weka.filters.MultiFilter();
            multi.setFilters(filters);

            fc.setFilter(multi);
        } else {
            fc.setFilter(new Normalize());
        }

        fc.setClassifier(factory.build());
        return fc;
    }

    private ModelResult eval(
            String name,
            boolean smote,
            Classifier cvModel,
            Instances data,
            Factory baseFactory) throws Exception {

        Evaluation ev = new Evaluation(data);
        ev.crossValidateModel(
                cvModel,
                data,
                Util.FOLDS,
                new Random(Util.SEED)
        );

        ModelResult r = new ModelResult();
        r.name = name;
        r.smote = smote;

        r.acc = ev.pctCorrect() / 100.0;

        int healthy = data.classAttribute().indexOfValue("1");

        // Safety fallback if the class labels are not exactly {0,1}.
        if (healthy < 0) {
            healthy = data.classAttribute().numValues() - 1;
        }

        r.prec = ev.precision(healthy);
        r.rec = ev.recall(healthy);
        r.f1 = ev.fMeasure(healthy);
        r.auc = ev.areaUnderROC(healthy);

        double[][] m = ev.confusionMatrix();
        r.cm = new int[m.length][];

        for (int i = 0; i < m.length; i++) {
            r.cm[i] = new int[m.length];
            for (int j = 0; j < m.length; j++) {
                r.cm[i][j] = (int) m[i][j];
            }
        }

        r.detail =
                ev.toSummaryString()
                        + "\n"
                        + ev.toClassDetailsString()
                        + "\n"
                        + ev.toMatrixString();

        // Final model trained on the complete dataset.
        r.fullModel = wrap(baseFactory, smote);
        r.fullModel.buildClassifier(data);

        r.importance = importance(
                baseFactory,
                smote,
                data,
                r.acc
        );

        return r;
    }

    /**
     * Permutation importance:
     * accuracy drop after shuffling one non-class attribute.
     */
    public Map<String, Double> importance(
            Factory base,
            boolean smote,
            Instances data,
            double baseline) throws Exception {

        Map<String, Double> out = new LinkedHashMap<>();

        for (int a = 0; a < data.numAttributes(); a++) {

            if (a == data.classIndex()) {
                continue;
            }

            Instances p = new Instances(data);

            int[] idx = shuffleIdx(p.numInstances());

            for (int i = 0; i < p.numInstances(); i++) {
                p.instance(i).setValue(
                        a,
                        data.instance(idx[i]).value(a)
                );
            }

            Evaluation ev = new Evaluation(p);

            ev.crossValidateModel(
                    wrap(base, smote),
                    p,
                    Util.FOLDS,
                    new Random(Util.SEED)
            );

            double permutedAccuracy = ev.pctCorrect() / 100.0;

            out.put(
                    data.attribute(a).name(),
                    baseline - permutedAccuracy
            );
        }

        List<Map.Entry<String, Double>> sorted =
                new ArrayList<>(out.entrySet());

        sorted.sort(
                (x, y) -> Double.compare(
                        y.getValue(),
                        x.getValue()
                )
        );

        out.clear();

        for (Map.Entry<String, Double> e : sorted) {
            out.put(e.getKey(), e.getValue());
        }

        return out;
    }

    private int[] shuffleIdx(int n) {
        int[] x = new int[n];

        for (int i = 0; i < n; i++) {
            x[i] = i;
        }

        Random r = new Random(Util.SEED);

        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);

            int t = x[i];
            x[i] = x[j];
            x[j] = t;
        }

        return x;
    }

    public void writeResults(List<ModelResult> rs) {

        StringBuilder cmp = new StringBuilder(
                "Model,SMOTE,Accuracy,Precision(H),Recall(H),F1(H),ROC-AUC(H)\n"
        );

        StringBuilder cms = new StringBuilder(
                "CONFUSION MATRICES (rows=actual, cols=predicted 0=NotHealthy 1=Healthy)\n"
        );

        for (ModelResult r : rs) {

            cmp.append(r.name)
                    .append(',')
                    .append(r.smote)
                    .append(',')
                    .append(Util.fmt(r.acc))
                    .append(',')
                    .append(Util.fmt(r.prec))
                    .append(',')
                    .append(Util.fmt(r.rec))
                    .append(',')
                    .append(Util.fmt(r.f1))
                    .append(',')
                    .append(Util.fmt(r.auc))
                    .append('\n');

            Util.writeFile(
                    "results/classification/"
                            + r.name.toLowerCase()
                            + (r.smote ? "_smote" : "_baseline")
                            + "_results.txt",
                    r.detail
            );

            cms.append("\n")
                    .append(r.name)
                    .append(r.smote ? " (SMOTE)" : " (baseline)")
                    .append("\n  TN=")
                    .append(r.cm[0][0])
                    .append(" FP=")
                    .append(r.cm[0][1])
                    .append(" FN=")
                    .append(r.cm[1][0])
                    .append(" TP=")
                    .append(r.cm[1][1])
                    .append('\n');
        }

        Util.writeFile(
                "results/classification/model_comparison.csv",
                cmp.toString()
        );

        Util.writeFile(
                "results/classification/confusion_matrices.txt",
                cms.toString()
        );

        for (ModelResult r : rs) {

            StringBuilder imp = new StringBuilder(
                    "Permutation importance (accuracy drop), "
                            + r.name
                            + (r.smote ? " +SMOTE" : " baseline")
                            + ":\n"
            );

            r.importance.forEach(
                    (k, v) -> imp.append(
                            String.format(
                                    "  %-18s %.4f%n",
                                    k,
                                    v
                            )
                    )
            );

            Util.writeFile(
                    "results/classification/importance_"
                            + r.name.toLowerCase()
                            + (r.smote ? "_smote" : "_baseline")
                            + ".txt",
                    imp.toString()
            );
        }

        // Best J48 variant for interpretability.
        rs.stream()
                .filter(r -> r.name.equals("J48"))
                .max(Comparator.comparingDouble(r -> r.f1))
                .ifPresent(r -> {
                    try {
                        Util.writeFile(
                                "results/classification/j48_tree.txt",
                                r.fullModel.toString()
                        );
                    } catch (Exception ignored) {
                    }
                });
    }
}
