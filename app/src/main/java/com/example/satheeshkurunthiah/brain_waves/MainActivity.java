package com.example.satheeshkurunthiah.brain_waves;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import static weka.core.SerializationHelper.read;
import static weka.core.SerializationHelper.write;


public class MainActivity extends AppCompatActivity {
    private static final String MODEL_NAME = "EEG_Dataset.model";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final Button trainButton = findViewById(R.id.trainButton);
        trainButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Spinner spinner = findViewById(R.id.algorithm);
                String algorithm = spinner.getSelectedItem().toString();

                // Train your algorithm based on this value
                if (algorithm.equals("SVM")) {
                    try {
                        long start = System.currentTimeMillis();
                        trainSVM();
                        long end = System.currentTimeMillis();
                        long totalTimeInSec = (end - start) / 1000;
                        print("Model successfully built and saved\nTook " + String.valueOf(totalTimeInSec) + " seconds");
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                System.out.print(algorithm);
            }
        });

        final Button testButton = findViewById(R.id.testButton);
        testButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                long start = System.currentTimeMillis();
                Spinner spinner = findViewById(R.id.inputFiles);
                String file = spinner.getSelectedItem().toString();

                // Test your algorithm based on this value
                DataSource source = null;
                if (file.equals("Random User SVM")) {
                    source = new DataSource(getResources().openRawResource(R.raw.random_user_svm));
                } else if (file.equals("Valid User SVM")) {
                    source = new DataSource(getResources().openRawResource(R.raw.valid_user_svm));
                }

                try {
                    if (source != null) {
                        String user = testSvm(source, file);
                        String message = "You are not Authorized";
                        if (user.equals("1")) {
                            message = "Authorized..!!";
                        }
                        long end = System.currentTimeMillis();
                        long totalTimeInSec = (end - start) / 1000;
                        print(message + "\nTook " + String.valueOf(totalTimeInSec) + " seconds");
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

        final Button performanceButton = findViewById(R.id.showPerformanceButton);
        performanceButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                // Code for displaying performance

                System.out.print("show performance");
            }
        });


    }

    private void trainSVM() throws Exception {
        DataSource source = new DataSource(getResources().openRawResource(R.raw.eeg_brain_svm));
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        Remove remove = new Remove();
        remove.setInputFormat(data);
        Instances newData = Filter.useFilter(data, remove);
        SMO scheme = new SMO();
        scheme.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
        scheme.buildClassifier(newData);
        write(getFilesDir() + "/" + MODEL_NAME, scheme);
    }

    private String testSvm(DataSource source, String inputFile) throws Exception {
        Instances unlabeled = new Instances(source.getDataSet());
        Classifier model = (Classifier) read(getFilesDir() + "/" + MODEL_NAME);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        Instances labeled = new Instances(unlabeled);
        Map<String, Integer> map = new HashMap<>();

        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = model.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
            String label = unlabeled.classAttribute().value((int) clsLabel);
            if (!map.containsKey(label)) {
                map.put(label, 0);
            }
            map.put(label, map.get(label) + 1);
        }

        BufferedWriter writer = new BufferedWriter(new FileWriter(getFilesDir() + "/" + inputFile + "_output.arff"));
        String output = labeled.toString();
        writer.write(output);
        writer.newLine();
        writer.flush();
        writer.close();
        Set<Map.Entry<String, Integer>> tree = new TreeSet<>(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
        tree.addAll(map.entrySet());

        return tree.iterator().next().getKey();
    }

    private void print(String message) {
        Toast toast = Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT);
        toast.show();
    }
}
