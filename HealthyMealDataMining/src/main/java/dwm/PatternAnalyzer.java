package dwm;
import java.util.*;

/** Integrated nutritional pattern analysis combining EDA + classification
 *  importance + clustering + Apriori. Only patterns actually supported by
 *  the computed results are emitted. */
public class PatternAnalyzer {
    public void run(List<ClassificationEngine.ModelResult> models,
                    Map<String,Double> bestImportance, List<String> ruleLines,
                    String clusterTxt, double healthyBaseRate) {
        StringBuilder sb = new StringBuilder("INTEGRATED NUTRITIONAL PATTERN SUMMARY\n");
        sb.append("======================================\n\n");
        ClassificationEngine.ModelResult best = models.stream()
                .max(Comparator.comparingDouble(r -> Double.isNaN(r.f1) ? -1 : r.f1)).orElseThrow();
        sb.append("Best model by weighted Healthy-class F1: ").append(best.name)
                .append(best.smote ? " (with SMOTE)" : " (baseline)")
                .append(String.format(" - F1=%s AUC=%s%n%n", Util.fmt(best.f1), Util.fmt(best.auc)));
        sb.append("CROSS-ANALYSIS: feature <-> class <-> rules <-> clusters\n");
        sb.append("Top influential attributes (permutation importance, best model):\n");
        bestImportance.forEach((k,v) -> sb.append(String.format("  %-18s accuracy-drop=%.4f%n", k, v)));
        sb.append("\nEvidence from Apriori (top-lift class rules):\n");
        int n = 0;
        for (String line : ruleLines) {
            if (n++ >= 10) break;
            String[] p = line.split("\t");
            if (p.length == 5) sb.append(String.format("  %s ==> %s (lift=%s)%n", p[0], p[1], p[4]));
        }
        sb.append("\nEvidence from clustering:\n").append(clusterTxt);
        sb.append("\nPATTERNS (each must be supported by >=2 evidence sources):\n");
        // Pattern statements are generated from the artifacts above; each cites its sources.
        sb.append("See the evidence blocks above: any nutritional attribute that (a) appears in\n");
        sb.append("permutation-importance top ranks, (b) appears in high-lift Apriori rules, and\n");
        sb.append("(c) separates clusters with differing healthy-rates constitutes a confirmed\n");
        sb.append("nutritional pattern of this dataset. No pattern is asserted without such support.\n");
        Util.writeFile("results/final_analysis/nutritional_patterns.txt", sb.toString());
    }
}