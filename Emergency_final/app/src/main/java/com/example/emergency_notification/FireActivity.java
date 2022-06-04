package com.example.emergency_notification;

import androidx.appcompat.app.AppCompatActivity;

import android.media.MediaPlayer;
import android.os.Bundle;
import android.widget.ImageView;

import com.bumptech.glide.Glide;

public class FireActivity extends AppCompatActivity {
    ImageView imageView5;
    MediaPlayer mediaPlayer;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_fire);

        imageView5 = (ImageView)findViewById(R.id.imageView5);
        Glide.with(this).load(R.raw.boxloading).into(imageView5);

        mediaPlayer = MediaPlayer.create(this, R.raw.danger1);
        mediaPlayer.setLooping(true); //무한재생
        mediaPlayer.start();

    }
}