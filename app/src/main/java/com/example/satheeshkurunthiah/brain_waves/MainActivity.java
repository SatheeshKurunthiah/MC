package com.example.satheeshkurunthiah.brain_waves;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;
import weka.core.Instances;

public class MainActivity extends AppCompatActivity {
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

                System.out.print(algorithm);
            }
        });

        final Button testButton = findViewById(R.id.testButton);
        testButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Spinner spinner = findViewById(R.id.inputFiles);
                String algorithm = spinner.getSelectedItem().toString();

                // Test your algorithm based on this value

                System.out.print(algorithm);
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

    public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
        Instances[][] split = new Instances[2][numberOfFolds];

        for (int i = 0; i < numberOfFolds; i++) {
            split[0][i] = data.trainCV(numberOfFolds, i);
            split[1][i] = data.testCV(numberOfFolds, i);
        }

        return split;
    }
}
