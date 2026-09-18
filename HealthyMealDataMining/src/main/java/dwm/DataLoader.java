package dwm;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

/**
 * Loads the raw Kaggle-style CSV into WEKA {@link Instances} and persists
 * the converted ARFF file, as required by the report (CSV -> ARFF conversion
 * through the WEKA Java API).
 */
public class DataLoader {

    /**
     * Loads a CSV file. Fails gracefully with a meaningful message when the
     * file is missing or malformed.
     */
    public Instances loadCSV(String path) {
        File f = new File(path);
        if (!f.exists()) {
            throw new IllegalArgumentException("Dataset file not found: " + path
                    + ". Place healthy_meal_plans.csv under the data/ directory.");
        }
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(f);
            loader.setMissingValue("?");
            Instances data = loader.getDataSet();
            if (data.numInstances() == 0 || data.numAttributes() == 0) {
                throw new IllegalArgumentException("Dataset " + path + " is empty or malformed.");
            }
            return data;
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to parse CSV " + path + ": " + e.getMessage(), e);
        }
    }

    /** Saves WEKA instances as an ARFF file, preserving names/types/values. */
    public void saveARFF(Instances data, String path) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(path));
            saver.writeBatch();
        } catch (Exception e) {
            throw new RuntimeException("Cannot write ARFF " + path + ": " + e.getMessage(), e);
        }
    }
}
