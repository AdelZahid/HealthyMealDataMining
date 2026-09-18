package dwm;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.*;

/**
 * Exploratory Data Analysis:
 * descriptive statistics, class distribution,
 * Pearson correlations, IQR outliers,
 * and per-meal-type healthiness.
 */
public class EDAAnalyzer {

    public void run(
            Instances master,
            DataInspector.Report rep,
            String mealAttrName) {

        StringBuilder txt =
                new StringBuilder(
                        "EXPLORATORY DATA ANALYSIS\n"
                                + "=========================\n"
                );

        StringBuilder statsCsv =
                new StringBuilder(
                        "attribute,type,count,mean,std,min,median,max\n"
                );

        List<double[]> numericCols =
                new ArrayList<>();

        List<String> numericNames =
                new ArrayList<>();

        for (int a = 0;
             a < master.numAttributes();
             a++) {

            Attribute at =
                    master.attribute(a);

            if (!at.isNumeric()
                    || a == master.classIndex()) {
                continue;
            }

            double[] v =
                    master.attributeToDoubleArray(a);

            numericCols.add(v);
            numericNames.add(at.name());

            double mean = 0.0;

            for (double x : v) {
                mean += x;
            }

            mean /= v.length;

            double var = 0.0;

            for (double x : v) {
                var +=
                        (x - mean)
                                * (x - mean);
            }

            var /= v.length;

            double[] s =
                    v.clone();

            Arrays.sort(s);

            statsCsv
                    .append(at.name())
                    .append(",numeric,")
                    .append(v.length)
                    .append(',')
                    .append(Util.fmt(mean))
                    .append(',')
                    .append(Util.fmt(Math.sqrt(var)))
                    .append(',')
                    .append(Util.fmt(s[0]))
                    .append(',')
                    .append(Util.fmt(s[s.length / 2]))
                    .append(',')
                    .append(Util.fmt(s[s.length - 1]))
                    .append('\n');
        }

        /*
         * Weka 3.8.x Utils.correlation() takes
         * three arguments:
         * correlation(x, y, n).
         */
        txt.append(
                "Pearson correlations among numeric attributes:\n"
        );

        for (int i = 0;
             i < numericNames.size();
             i++) {

            for (int j = i + 1;
                 j < numericNames.size();
                 j++) {

                double[] x =
                        numericCols.get(i);

                double[] y =
                        numericCols.get(j);

                double c =
                        weka.core.Utils.correlation(
                                x,
                                y,
                                Math.min(
                                        x.length,
                                        y.length
                                )
                        );

                if (Math.abs(c) >= 0.3) {

                    txt.append(
                            String.format(
                                    "  %s vs %s : %.3f%n",
                                    numericNames.get(i),
                                    numericNames.get(j),
                                    c
                            )
                    );
                }
            }
        }

        // IQR outlier analysis.
        txt.append(
                "\nPotential outliers "
                        + "(IQR rule, reported not removed):\n"
        );

        int outlierCount = 0;

        for (int i = 0;
             i < numericNames.size();
             i++) {

            double[] s =
                    numericCols.get(i).clone();

            Arrays.sort(s);

            double q1 =
                    s[s.length / 4];

            double q3 =
                    s[(3 * s.length) / 4];

            double iqr =
                    q3 - q1;

            double lo =
                    q1 - 1.5 * iqr;

            double hi =
                    q3 + 1.5 * iqr;

            int n = 0;

            for (double x :
                    numericCols.get(i)) {

                if (x < lo || x > hi) {
                    n++;
                }
            }

            txt.append(
                    String.format(
                            "  %-16s %d values outside [%.3f, %.3f]%n",
                            numericNames.get(i),
                            n,
                            lo,
                            hi
                    )
            );

            outlierCount += n;
        }

        txt.append(
                "Total outlier values: "
                        + outlierCount
                        + " - retained (they may represent "
                        + "legitimately extreme meals).\n"
        );

        // Healthiness rate by meal type.
        int mealIdx =
                master.attribute(
                        mealAttrName
                ).index();

        Map<String, int[]> byMeal =
                new TreeMap<>();

        for (int i = 0;
             i < master.numInstances();
             i++) {

            String m =
                    master.instance(i)
                            .stringValue(mealIdx);

            int h =
                    (int) master.instance(i)
                            .classValue();

            byMeal
                    .computeIfAbsent(
                            m,
                            k -> new int[2]
                    )[h]++;
        }

        txt.append(
                "\nHealthiness rate per meal type:\n"
        );

        for (Map.Entry<String, int[]> e :
                byMeal.entrySet()) {

            int healthy =
                    e.getValue()[1];

            int total =
                    e.getValue()[0]
                            + e.getValue()[1];

            txt.append(
                    String.format(
                            "  %-22s %2d/%2d healthy (%.1f%%)%n",
                            e.getKey(),
                            healthy,
                            total,
                            100.0 * healthy / total
                    )
            );
        }

        Util.writeFile(
                "results/eda/descriptive_statistics.csv",
                statsCsv.toString()
        );

        Util.writeFile(
                "results/eda/nutritional_analysis.txt",
                txt.toString()
        );

        StringBuilder cd =
                new StringBuilder(
                        "class,count,percent\n"
                );

        for (int v = 0;
             v < rep.classLabels.length;
             v++) {

            cd.append(
                    rep.classLabels[v]
            )
                    .append(',')
                    .append(rep.classCounts[v])
                    .append(',')
                    .append(
                            Util.F2.format(
                                    100.0
                                            * rep.classCounts[v]
                                            / rep.rows
                            )
                    )
                    .append('\n');
        }

        Util.writeFile(
                "results/eda/class_distribution.csv",
                cd.toString()
        );
    }
}
