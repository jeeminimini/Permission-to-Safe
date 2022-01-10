package com.example.cs_project;


import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.android.volley.AuthFailureError;
import com.android.volley.NetworkResponse;
import com.android.volley.ParseError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.HttpHeaderParser;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    //Volley 라이브러리 사용하여 웹 요청과 응답 처리하기
    //안드로이드는 요청응답을 스레드로 돌려야 하지만 Volley 라이브러리를 사용하면 시경 쓰지 않아도 된다.

    //main.xml 레이아웃에 설정된 뷰들을 가져오는 메소드 = findViewById
    //->이벤트 처리가 가능하게 된다
    //먼저 요청(Request) 객체를 만들고 이 요청 객체를 요청 큐(RequestQueue)라는 곳에 넣어주기만 하면 됩니다.
    //그러면 요청 큐가 알아서 웹서버에 요청하고 응답까지 받아 사용자가 사용할 수 있도록 지정된 메소드를 호출해줍니다.
    //응답이 오면 처리해주는건 request객체의 Listener에 구현

    JSONArray data_list=null;

    TextView text;
    Button button,btn;

    RequestQueue requestQueue=null; //요청 큐는 한 번만 만들어 계속 사용 : static

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //id mapping
        text=(TextView) findViewById(R.id.text);
        //id mapping
        button=(Button) findViewById(R.id.button);
        btn=(Button)findViewById(R.id.btn);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendRequest();
            }
        });


        if(requestQueue == null){

            //리퀘스트큐 생성 (MainActivit가 메모리에서 만들어질 때 같이 생성이 될것이다.
            requestQueue = Volley.newRequestQueue(getApplicationContext());
        }
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this,Question_os.class);
                startActivity(intent);
            }
        });
    }
    public void sendRequest(){
        String url = "http://172.30.1.60:5000/return_os_quiz";

        //StringRequest를 만듬 (파라미터구분을 쉽게하기위해 엔터를 쳐서 구분하면 좋다)
        //StringRequest는 요청객체중 하나이며 가장 많이 쓰인다고한다.
        //요청객체는 다음고 같이 보내는방식(GET,POST), URL, 응답성공리스너, 응답실패리스너 이렇게 4개의 파라미터를 전달할 수 있다.(리퀘스트큐에 ㅇㅇ)
        //화면에 결과를 표시할때 핸들러를 사용하지 않아도되는 장점이있다.
        StringRequest request = new StringRequest(
                Request.Method.GET,
                url,
                new Response.Listener<String>() {  //응답을 문자열로 받아서 여기다 넣어달란말임(응답을 성공적으로 받았을 떄 이메소드가 자동으로 호출됨
                    @Override
                    public void onResponse(String response) {
                        //응답이 response로 오고 난 후
                        //data_list=makeJSON(response);
                        //print_jsonArray(data_list);
                        //println(response);
                    }
                },
                new Response.ErrorListener(){ //에러발생시 호출될 리스너 객체
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        println("에러 => "+ error.getMessage());
                    }
                }
        ){

            //response를 UTF8로 변경해주는 소스코드 시작
            @Override //response를 UTF8로 변경해주는 소스코드
            protected Response<String> parseNetworkResponse(NetworkResponse response) {
                try {
                    String utf8String = new String(response.data, "UTF-8");
                    return Response.success(utf8String, HttpHeaderParser.parseCacheHeaders(response));
                } catch (UnsupportedEncodingException e) {
                    // log error
                    return Response.error(new ParseError(e));
                } catch (Exception e) {
                    // log error
                    return Response.error(new ParseError(e));
                }
            }
            //response를 UTF8로 변경해주는 소스코드 끝

            @Override
            protected Map<String, String> getParams() throws AuthFailureError {
                Map<String,String> params = new HashMap<String,String>();

                return params;
            }


        };

        //아래 add코드처럼 넣어줄때 Volley라고하는게 내부에서 캐싱을 해준다, 즉, 한번 보내고 받은 응답결과가 있으면
        //그 다음에 보냈을 떄 이전 게 있으면 그냥 이전거를 보여줄수도  있다.
        //따라서 이렇게 하지말고 매번 받은 결과를 그대로 보여주기 위해 다음과같이 setShouldCache를 false로한다.
        //결과적으로 이전 결과가 있어도 새로 요청한 응답을 보여줌
        request.setShouldCache(false); //이전 결과 있어도 새로 요청하여 응답 보여준다

        requestQueue.add(request);

        println("요청 보냄.");
    }

    public JSONArray makeJSON(String res){
        //응답 받은 json 문자열을 jsonarray로 가공하는 함수
        try {

            // String 으로 들어온 값 JSONArray로 파싱
            JSONArray data_list = new JSONArray(res);

        } catch (JSONException e) {
            e.printStackTrace();
        }

        return data_list;
    }

    public void print_jsonArray(JSONArray data_list){

        String question=null;
        String ans_1=null;
        String ans_2=null;
        String ans_3=null;
        String ans_4=null;
        String ans=null;

        try{
            for(int i=0;i<2;i++){
                JSONObject data=data_list.getJSONObject(i);
                question=data.getString("question");
                ans_1=data.getString("ans_1");
                ans_2=data.getString("ans_2");
                ans_3=data.getString("ans_3");
                ans_4=data.getString("ans_4");
                ans=data.getString("ans");

            }

        }catch (JSONException e) {
            e.printStackTrace();
        }

        //첫번째 정보만 우선 출력해보기
        //String show_data=data.getString("str");
        String show_data="문제 : "+question+"\n1."+ans_1+"\n2."+ans_2+"\n3."+ans_3+"\n4."+ans_4+"\n"+"답 : "+ans;

        //System.out.println("이거 문제보여주는거임\n\n"+show_data+"이거 문제보여주는거임\n\n");
        println(show_data);

    }

    public void println(String str){
        //text set

        text.setText(str+"\n");
    }















}