import weka.filters.supervised.instance.SMOTE;

public class TestSMOTE {
    public static void main(String[] args) {
        SMOTE smote = new SMOTE();
        smote.setRandomSeed(1);

        System.out.println("SMOTE loaded successfully!");
        System.out.println("Random seed: " + smote.getRandomSeed());
    }
}