package dwm;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.AssociationRulesProducer;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Discretized nutrition + Apriori.
 *
 * Rules are read through Weka's AssociationRule API.
 * High-lift rules with a class consequent are emphasised.
 */
public class AssociationRuleMiner {

    public List<String> analyses = new ArrayList<>();

    public void run(Instances aprioriData) throws Exception {

        Apriori ap = new Apriori();

        ap.setClassIndex(aprioriData.classIndex());

        ap.setOptions(new String[]{
                "-N", "60",
                "-T", "0",
                "-C", "0.6",
                "-D", "0.03",
                "-U", "1.0",
                "-M", "0.03",
                "-S", "-1.0",
        });

        ap.buildAssociations(aprioriData);

        AssociationRulesProducer prod = ap;

        List<String> all =
                new StringBuilderRules()
                        .collect(prod.getAssociationRules(), aprioriData);

        Util.writeFile(
                "results/association_rules/apriori_rules.txt",
                ap.toString()
                        + "\n\nPARSED RULES:\n"
                        + String.join("\n", all)
        );

        Util.writeFile(
                "results/association_rules/important_rules.csv",
                "antecedent,consequent,support,confidence,lift\n"
                        + String.join("\n", all)
        );

        StringBuilder ana = new StringBuilder(
                "ASSOCIATION RULE ANALYSIS "
                        + "(Apriori, min support 0.03, min confidence 0.6)\n"
        );

        ana.append(
                "Discretization: equal-width, 3 bins "
                        + "(Low/Medium/High) for num_ingredients, calories, "
                        + "prep_time, protein, fat, carbs.\n"
        );

        ana.append(
                "Diet flags kept as nominal 0/1. "
                        + "Class values: 0=Not Healthy, 1=Healthy.\n"
        );

        ana.append(
                "Interpretation uses 'associated with' - association rules "
                        + "show co-occurrence, not causation.\n\n"
        );

        int shown = 0;

        for (String line : all) {

            if (shown >= 15) {
                break;
            }

            String[] p = line.split("\t");

            if (p.length == 5 && p[1].contains("is_healthy")) {

                ana.append("Rule: ")
                        .append(p[0])
                        .append("  ==>  ")
                        .append(p[1])
                        .append(String.format(
                                "%n  support=%s confidence=%s lift=%s%n",
                                p[2],
                                p[3],
                                p[4]
                        ));

                ana.append(
                        "  Interpretation: meals with this nutritional "
                                + "combination occur together with class '"
                )
                        .append(
                                p[1].contains("=1")
                                        ? "Healthy"
                                        : "Not Healthy"
                        )
                        .append(
                                "' at lift "
                                        + p[4]
                                        + " times the base rate - an "
                                        + "association, not proof of causation.%n%n"
                        );

                shown++;
            }
        }

        analyses.add(ana.toString());

        Util.writeFile(
                "results/association_rules/rule_analysis.txt",
                ana.toString()
        );
    }

    static class StringBuilderRules {

        List<String> collect(
                AssociationRules rules,
                Instances data) throws Exception {

            List<String> out = new ArrayList<>();

            List<AssociationRule> list =
                    rules.getRules();

            for (AssociationRule r : list) {

                /*
                 * Weka 3.8.x uses getNamedMetricValue("Lift"),
                 * not getMetricValuesForName("Lift").
                 */
                double lift;

                try {
                    lift = r.getNamedMetricValue("Lift");
                } catch (Exception e) {
                    lift = findMetric(
                            r,
                            "lift"
                    );
                }

                /*
                 * getPremise() and getConsequence() return
                 * Collection<Item>, not List<Item>.
                 */
                out.add(
                        items(r.getPremise())
                                + "\t"
                                + items(r.getConsequence())
                                + "\t"
                                + r.getTotalSupport()
                                + "\t"
                                + Util.fmt(r.getPrimaryMetricValue())
                                + "\t"
                                + Util.fmt(lift)
                );
            }

            out.sort(
                    (x, y) ->
                            Double.compare(
                                    parseLift(y),
                                    parseLift(x)
                            )
            );

            return out;
        }

        private double findMetric(
                AssociationRule rule,
                String wanted) throws Exception {

            String[] names =
                    rule.getMetricNamesForRule();

            double[] values =
                    rule.getMetricValuesForRule();

            for (int i = 0; i < names.length; i++) {
                if (names[i].equalsIgnoreCase(wanted)) {
                    return values[i];
                }
            }

            return Double.NaN;
        }

        private double parseLift(String line) {

            try {
                return Double.parseDouble(
                        line.split("\t")[4]
                );
            } catch (Exception e) {
                return 0.0;
            }
        }

        private String items(Collection<?> items) {

            StringBuilder sb =
                    new StringBuilder();

            for (Object o : items) {

                if (sb.length() > 0) {
                    sb.append(" & ");
                }

                sb.append(o);
            }

            return sb.toString();
        }
    }
}
