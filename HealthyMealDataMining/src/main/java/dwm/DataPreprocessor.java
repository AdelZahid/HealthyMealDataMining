package dwm;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.io.File;

/**
 * Implements the preprocessing steps required by the report.
 *
 * ReplaceMissingValues -> StringToNominal -> NumericToNominal(class)
 *
 * Task-specific views are then created for classification, clustering,
 * and Apriori.
 */
public class DataPreprocessor {

    /** Applies a Weka filter and returns the filtered dataset. */
    public static Instances apply(
            Filter filter,
            Instances data) throws Exception {

        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    /**
     * Master preprocessing chain.
     */
    public Instances master(Instances raw) throws Exception {

        Instances d = raw;

        // 1. Replace missing values.
        d = apply(
                new ReplaceMissingValues(),
                d
        );

        // 2. Convert meal_name from String to Nominal.
        StringToNominal stn =
                new StringToNominal();

        stn.setAttributeRange("first");

        d = apply(stn, d);

        // 3. Convert target from numeric 0/1 to nominal.
        NumericToNominal ntn =
                new NumericToNominal();

        // NumericToNominal uses setAttributeIndices(), not
        // StringToNominal's setAttributeRange().
        ntn.setAttributeIndices("last");

        d = apply(ntn, d);

        d.setClassIndex(
                d.numAttributes() - 1
        );

        return d;
    }

    /**
     * Classification view:
     * removes meal_name because it is an identifier.
     */
    public Instances classificationView(
            Instances master) throws Exception {

        Remove rm = new Remove();

        rm.setAttributeIndices("1");

        Instances d =
                apply(rm, master);

        d.setClassIndex(
                d.numAttributes() - 1
        );

        return d;
    }

    /**
     * Clustering view:
     * removes meal_name and the class,
     * then applies normalization.
     */
    public Instances clusteringView(
            Instances master) throws Exception {

        Remove rm = new Remove();

        rm.setAttributeIndices(
                "1," + (master.classIndex() + 1)
        );

        Instances d =
                apply(rm, master);

        Normalize norm =
                new Normalize();

        d = apply(norm, d);

        return d;
    }

    /**
     * Apriori view:
     *
     * 1. Remove meal_name.
     * 2. Convert diet flags to nominal.
     * 3. Discretize six nutritional numeric attributes
     *    into three bins.
     */
    public Instances aprioriView(
            Instances master) throws Exception {

        Remove rm = new Remove();

        rm.setAttributeIndices("1");

        Instances d =
                apply(rm, master);

        /*
         * After removing meal_name:
         *
         * 1..6  = nutritional numeric attributes
         * 7..12 = diet flags
         * 13    = is_healthy
         */
        NumericToNominal ntn =
                new NumericToNominal();

        ntn.setAttributeIndices("7-12");

        d = apply(ntn, d);

        d.setClassIndex(
                d.numAttributes() - 1
        );

        Discretize disc =
                new Discretize();

        disc.setAttributeIndices("1-6");

        /*
         * Weka 3.8.x does not provide setUseEqualWidth().
         * The default Discretize behavior is used here and
         * the number of bins is explicitly fixed to 3.
         */
        disc.setBins(3);

        d = apply(disc, d);

        return d;
    }

    /** Saves an Instances object as ARFF. */
    public void saveARFF(
            Instances data,
            String path) {

        try {

            ArffSaver saver =
                    new ArffSaver();

            saver.setInstances(data);

            saver.setFile(
                    new File(path)
            );

            saver.writeBatch();

        } catch (Exception e) {

            throw new RuntimeException(
                    "ARFF save failed: "
                            + e.getMessage(),
                    e
            );
        }
    }
}
