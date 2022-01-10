package com.example.cs_project;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;


public class answer extends AppCompatActivity {
    TextView an;
    Button button;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_answer);
        an=(TextView) findViewById(R.id.ans);
        Intent intent = getIntent();
        String Result = intent.getStringExtra("tot");
        an.setText(Result);
        /*int []check=intent.getIntArrayExtra("check");
        System.out.println("길이"+check.length);
        String str="";
        for(int i=0;i<10;i++) {
            str+=String.valueOf(check[i]);

        }
        System.out.println("체크체크:"+str);*/
        button=(Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(),HomeActivity.class);
                startActivity(intent);
            }
        });
    }
}


