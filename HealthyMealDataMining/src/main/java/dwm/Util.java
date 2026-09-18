package dwm;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.List;

/**
 * Shared constants and small helpers.
 *
 * <p>Reproducibility note: every stochastic component (RandomForest, SMO
 * internal randomisation, K-Means, Apriori item ordering, fold assignment,
 * permutation importance) uses the fixed {@link #SEED} below.
 */
public final class Util {

    /** Fixed random seed used across the whole pipeline. */
    public static final int SEED = 1;

    /** Cross-validation folds (stratified 10-fold, as required by the report). */
    public static final int FOLDS = 10;

    public static final DecimalFormat F4 = new DecimalFormat("0.0000");
    public static final DecimalFormat F2 = new DecimalFormat("0.00");

    public static final String CSV_PATH = "data/healthy_meal_plans.csv";
    public static final String ARFF_PATH = "data/healthy_meal_plans.arff";

    private Util() { }

    public static void ensureDirs() {
        String[] dirs = {
                "models",
                "results/preprocessing", "results/eda", "results/classification",
                "results/clustering", "results/association_rules", "results/final_analysis"
        };
        for (String d : dirs) {
            new File(d).mkdirs();
        }
    }

    public static void writeFile(String path, String content) {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.print(content);
        } catch (IOException e) {
            throw new RuntimeException("Cannot write " + path + ": " + e.getMessage(), e);
        }
    }

    /** Map the nominal class value "1"/"0" to a human readable label. */
    public static String healthyLabel(double value) {
        return value == 1.0 ? "Healthy" : "Not Healthy";
    }

    public static String fmt(double v) {
        if (Double.isNaN(v)) {
            return "n/a";
        }
        return F4.format(v);
    }

    public static String csvJoin(List<String> cells) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cells.size(); i++) {
            String c = cells.get(i);
            if (c.contains(",") || c.contains("\"")) {
                c = "\"" + c.replace("\"", "\"\"") + "\"";
            }
            sb.append(c);
            if (i < cells.size() - 1) {
                sb.append(',');
            }
        }
        return sb.toString();
    }
}
