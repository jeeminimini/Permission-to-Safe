package com.example.emergency_notification;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.ImageView;

import com.bumptech.glide.Glide;

public class NormalActivity extends AppCompatActivity {

    ImageView imageView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_normal);

        imageView = (ImageView)findViewById(R.id.imageView);
        Glide.with(this).load(R.raw.smile_long).into(imageView);
    }
}