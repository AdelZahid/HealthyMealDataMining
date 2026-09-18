package dwm;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Programmatic dataset inspection: dimensions, attribute names/types,
 * missing values, duplicate records, unique values, class detection and
 * class distribution. Nothing here is hard-coded; everything is computed
 * from the actual CSV.
 */
public class DataInspector {

    /** Result holder for everything the inspector computes. */
    public static class Report {
        public Instances data;
        public int rows;
        public int cols;
        public int missing;
        public int duplicates;
        public int classIndex = -1;
        public String className = "";
        public int[] classCounts;
        public String[] classLabels;
        public double imbalanceRatio;
        public String summary;
    }

    public Report inspect(Instances data) {
        Report r = new Report();
        r.data = data;
        r.rows = data.numInstances();
        r.cols = data.numAttributes();

        int missing = 0;
        Set<String> seen = new HashSet<>();
        int duplicates = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            StringBuilder sig = new StringBuilder();
            for (int a = 0; a < data.numAttributes(); a++) {
                if (data.instance(i).isMissing(a)) {
                    missing++;
                }
                sig.append(data.instance(i).value(a)).append('|');
            }
            if (!seen.add(sig.toString())) {
                duplicates++;
            }
        }
        r.missing = missing;
        r.duplicates = duplicates;

        // Detect the class attribute by name (report specifies "Healthy").
        for (int a = 0; a < data.numAttributes(); a++) {
            String name = data.attribute(a).name().toLowerCase();
            if (name.equals("is_healthy") || name.equals("healthy")) {
                r.classIndex = a;
                break;
            }
        }
        if (r.classIndex < 0) {
            r.classIndex = data.numAttributes() - 1;
        }
        data.setClassIndex(r.classIndex);
        r.className = data.attribute(r.classIndex).name();

        Attribute cls = data.classAttribute();
        if (cls.isNominal()) {
            r.classCounts = new int[cls.numValues()];
            for (int i = 0; i < data.numInstances(); i++) {
                if (!data.instance(i).classIsMissing()) {
                    r.classCounts[(int) data.instance(i).classValue()]++;
                }
            }
            r.classLabels = new String[cls.numValues()];
            for (int v = 0; v < cls.numValues(); v++) {
                r.classLabels[v] = cls.value(v);
            }
        } else {
            // numeric class: count distinct rounded values
            Map<Double, Integer> counts = new HashMap<>();
            for (int i = 0; i < data.numInstances(); i++) {
                double v = data.instance(i).classValue();
                counts.merge(v, 1, Integer::sum);
            }
            r.classCounts = new int[counts.size()];
            r.classLabels = new String[counts.size()];
            int i = 0;
            for (Map.Entry<Double, Integer> e : counts.entrySet()) {
                r.classLabels[i] = String.valueOf(e.getKey());
                r.classCounts[i] = e.getValue();
                i++;
            }
        }
        int maj = 0, min = Integer.MAX_VALUE;
        for (int c : r.classCounts) {
            maj = Math.max(maj, c);
            min = Math.min(min, c);
        }
        r.imbalanceRatio = min == 0 ? Double.NaN : (double) maj / min;

        r.summary = buildSummary(r, data);
        return r;
    }

    private String buildSummary(Report r, Instances data) {
        StringBuilder sb = new StringBuilder();
        sb.append("DATASET INSPECTION SUMMARY\n");
        sb.append("==========================\n");
        sb.append("Source file           : ").append(Util.CSV_PATH).append('\n');
        sb.append("Rows (instances)      : ").append(r.rows).append('\n');
        sb.append("Columns (attributes)  : ").append(r.cols).append('\n');
        sb.append("Missing values        : ").append(r.missing).append('\n');
        sb.append("Duplicate records     : ").append(r.duplicates).append('\n');
        sb.append("Detected class        : ").append(r.className)
                .append(" (index ").append(r.classIndex).append(")\n");
        sb.append('\n');
        sb.append("Attributes:\n");
        sb.append(String.format("%-4s %-20s %-12s %-10s %-10s %s%n",
                "No.", "Name", "Type", "Distinct", "Missing", "Sample/Range"));
        for (int a = 0; a < data.numAttributes(); a++) {
            Attribute at = data.attribute(a);
            String type;
            if (at.isNominal()) {
                type = "nominal";
            } else if (at.isNumeric()) {
                type = "numeric";
            } else if (at.isString()) {
                type = "string";
            } else {
                type = "date";
            }
            int distinct = at.numValues(); // for nominal/string
            String range;
            if (at.isNumeric()) {
                try {
                    range = "[" + F(at, true) + " .. " + F(at, false) + "]";
                } catch (Exception e) {
                    range = "n/a";
                }
            } else {
                StringBuilder vals = new StringBuilder("{");
                int shown = Math.min(distinct, 6);
                for (int v = 0; v < shown; v++) {
                    vals.append(at.value(v));
                    if (v < shown - 1) {
                        vals.append(", ");
                    }
                }
                if (distinct > shown) {
                    vals.append(", ...");
                }
                vals.append('}');
                range = vals.toString();
            }
            int miss = 0;
            for (int i = 0; i < data.numInstances(); i++) {
                if (data.instance(i).isMissing(a)) {
                    miss++;
                }
            }
            sb.append(String.format("%-4d %-20s %-12s %-10d %-10d %s%n",
                    a + 1, at.name(), type, distinct, miss, range));
        }
        sb.append('\n');
        sb.append("Class distribution (").append(r.className).append("):\n");
        for (int v = 0; v < r.classLabels.length; v++) {
            double pct = 100.0 * r.classCounts[v] / r.rows;
            sb.append(String.format("  value %-8s count %-5d (%.2f%%)%n",
                    r.classLabels[v], r.classCounts[v], pct));
        }
        sb.append(String.format("%nClass imbalance ratio (majority/minority): %.2f : 1%n", r.imbalanceRatio));
        return sb.toString();
    }

    private String F(Attribute a, boolean min) {
        double v = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        for (int i = 0; i < a.numValues(); i++) {
            double x = Double.parseDouble(a.value(i));
            v = min ? Math.min(v, x) : Math.max(v, x);
        }
        return String.valueOf(v);
    }
}
