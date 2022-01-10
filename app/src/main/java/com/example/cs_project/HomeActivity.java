package com.example.cs_project;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class HomeActivity extends Activity {
    Button btn_os,btn_com,btn_sum;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);
        btn_os=(Button)findViewById(R.id.btn_os);
        btn_com=(Button)findViewById(R.id.btn_com);
        btn_sum=(Button)findViewById(R.id.btn_sum);

        btn_os.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(HomeActivity.this,Question_os.class);
                startActivity(intent1);
            }
        });

        btn_com.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent2 = new Intent(HomeActivity.this,Question_cs.class);
                startActivity(intent2);
            }
        });

        btn_sum.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent3 = new Intent(HomeActivity.this,Question_sum.class);
                startActivity(intent3);
            }
        });
    }
}
