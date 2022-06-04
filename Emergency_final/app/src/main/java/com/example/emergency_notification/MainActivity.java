package com.example.emergency_notification;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.net.UnknownHostException;

public class MainActivity extends Activity {
    Button btn_connection;
    Socket socket;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btn_connection = (Button) findViewById(R.id.btn_connection);
        socket = new Socket();

        btn_connection.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                MyClientTask myClientTask = new MyClientTask("10.0.2.2", 12000); //10.0.2.2
                myClientTask.execute();

            }
        });

    }
    public class MyClientTask extends AsyncTask<Void, Void, Void> {
        String dstAddress;
        int dstPort;
        String response;

        //constructor
        MyClientTask(String addr, int port){
            dstAddress = addr;//이게 호스트
            dstPort = port;//이게 포트
            String response="";
        }

        @Override
        protected Void doInBackground(Void... arg0) {

            Socket socket = null;
            try {
                socket = new Socket(dstAddress, dstPort);
                //송신
                String myMessage;
                OutputStream out = socket.getOutputStream();
                myMessage = "통신시작";
                out.write(myMessage.getBytes());
                //수신
                ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(1024);
                byte[] bytes = null;
                String clientMessage;
                InputStream inputStream = socket.getInputStream();
                bytes=new byte[1024];
                int readByteCount = inputStream.read(bytes);
                clientMessage= new String(bytes,0,readByteCount,"UTF-8");
                response = "현재상황 : "+clientMessage;
                /*
                 * notice:
                 * inputStream.read() will block if no data return
                 */

                //sendRequest(clientMessage);

            } catch (UnknownHostException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                response = "UnknownHostException: " + e.toString();
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
                response = "IOException: " + e.toString();
            }finally{
                if(socket != null){
                    try {
                        socket.close();
                    } catch (IOException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();

                    }
                }
            }
            return null;
        }
        // doInBackground( ) 메소드에서 작업이 끝나면
        // onPostExcuted( ) 로 결과 파라미터를 리턴하면서
        // 그 리턴값을 통해 스레드 작업이 끝났을 때의 동작 구현
        @Override
        protected void onPostExecute(Void result) {
            super.onPostExecute(result);
            if (response.equals("현재상황 : 위급상황")){
                Intent intent = new Intent(getApplicationContext(),FireActivity.class);
                startActivity(intent);
            }
            else if(response.equals("현재상황 : 일반상황")){
                Intent intent = new Intent(getApplicationContext(), NormalActivity.class);
                startActivity(intent);
            }
        }
    }
}
