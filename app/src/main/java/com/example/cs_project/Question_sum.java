package com.example.cs_project;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

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
import java.util.ArrayList;

public class Question_sum extends AppCompatActivity {
    int tot = 0;

    TextView textView1,textView2,textView3,textView4,textView5,textView6,textView7,textView8,textView9,textView10;
    RadioGroup radioGroup1,radioGroup2,radioGroup3,radioGroup4,radioGroup5,radioGroup6,radioGroup7,radioGroup8,radioGroup9,radioGroup10;
    RadioButton radio1_1, radio1_2, radio1_3, radio1_4,radio2_1, radio2_2, radio2_3, radio2_4,radio3_1, radio3_2, radio3_3, radio3_4,radio4_1, radio4_2, radio4_3, radio4_4,
            radio5_1, radio5_2, radio5_3, radio5_4,radio6_1, radio6_2, radio6_3, radio6_4,radio7_1, radio7_2, radio7_3, radio7_4,radio8_1, radio8_2, radio8_3, radio8_4,
            radio9_1, radio9_2, radio9_3, radio9_4,radio10_1, radio10_2, radio10_3, radio10_4;
    Button btn,btn_start;
    LinearLayout linearLayout1,linearLayout2,linearLayout3,linearLayout4,linearLayout5,linearLayout6,linearLayout7,linearLayout8,linearLayout9,linearLayout10;

    JSONArray data_list;
    public ArrayList<Data_jsonObject> data=new ArrayList<Data_jsonObject>();

    RequestQueue requestQueue=null; //요청 큐는 한 번만 만들어 계속 사용 : static

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_question_sum);

        btn=(Button)findViewById(R.id.btn);
        btn_start=(Button)findViewById(R.id.btn_start);

        textView1=(TextView) findViewById(R.id.textView1);
        radioGroup1 = (RadioGroup) findViewById(R.id.radioGroup1);
        radio1_1 = (RadioButton) findViewById(R.id.radio1_1);
        radio1_2 = (RadioButton) findViewById(R.id.radio1_2);
        radio1_3 = (RadioButton) findViewById(R.id.radio1_3);
        radio1_4 = (RadioButton) findViewById(R.id.radio1_4);

        textView2=(TextView) findViewById(R.id.textView2);
        radioGroup2 = (RadioGroup) findViewById(R.id.radioGroup2);
        radio2_1 = (RadioButton) findViewById(R.id.radio2_1);
        radio2_2 = (RadioButton) findViewById(R.id.radio2_2);
        radio2_3 = (RadioButton) findViewById(R.id.radio2_3);
        radio2_4 = (RadioButton) findViewById(R.id.radio2_4);

        textView3=(TextView) findViewById(R.id.textView3);
        radioGroup3 = (RadioGroup) findViewById(R.id.radioGroup3);
        radio3_1 = (RadioButton) findViewById(R.id.radio3_1);
        radio3_2 = (RadioButton) findViewById(R.id.radio3_2);
        radio3_3 = (RadioButton) findViewById(R.id.radio3_3);
        radio3_4 = (RadioButton) findViewById(R.id.radio3_4);

        textView4=(TextView) findViewById(R.id.textView4);
        radioGroup4 = (RadioGroup) findViewById(R.id.radioGroup4);
        radio4_1 = (RadioButton) findViewById(R.id.radio4_1);
        radio4_2 = (RadioButton) findViewById(R.id.radio4_2);
        radio4_3 = (RadioButton) findViewById(R.id.radio4_3);
        radio4_4 = (RadioButton) findViewById(R.id.radio4_4);

        textView5=(TextView) findViewById(R.id.textView5);
        radioGroup5 = (RadioGroup) findViewById(R.id.radioGroup5);
        radio5_1 = (RadioButton) findViewById(R.id.radio5_1);
        radio5_2 = (RadioButton) findViewById(R.id.radio5_2);
        radio5_3 = (RadioButton) findViewById(R.id.radio5_3);
        radio5_4 = (RadioButton) findViewById(R.id.radio5_4);

        textView6=(TextView) findViewById(R.id.textView6);
        radioGroup6 = (RadioGroup) findViewById(R.id.radioGroup6);
        radio6_1 = (RadioButton) findViewById(R.id.radio6_1);
        radio6_2 = (RadioButton) findViewById(R.id.radio6_2);
        radio6_3 = (RadioButton) findViewById(R.id.radio6_3);
        radio6_4 = (RadioButton) findViewById(R.id.radio6_4);

        textView7=(TextView) findViewById(R.id.textView7);
        radioGroup7 = (RadioGroup) findViewById(R.id.radioGroup7);
        radio7_1 = (RadioButton) findViewById(R.id.radio7_1);
        radio7_2 = (RadioButton) findViewById(R.id.radio7_2);
        radio7_3 = (RadioButton) findViewById(R.id.radio7_3);
        radio7_4 = (RadioButton) findViewById(R.id.radio7_4);

        textView8=(TextView) findViewById(R.id.textView8);
        radioGroup8 = (RadioGroup) findViewById(R.id.radioGroup8);
        radio8_1 = (RadioButton) findViewById(R.id.radio8_1);
        radio8_2 = (RadioButton) findViewById(R.id.radio8_2);
        radio8_3 = (RadioButton) findViewById(R.id.radio8_3);
        radio8_4 = (RadioButton) findViewById(R.id.radio8_4);

        textView9=(TextView) findViewById(R.id.textView9);
        radioGroup9 = (RadioGroup) findViewById(R.id.radioGroup9);
        radio9_1 = (RadioButton) findViewById(R.id.radio9_1);
        radio9_2 = (RadioButton) findViewById(R.id.radio9_2);
        radio9_3 = (RadioButton) findViewById(R.id.radio9_3);
        radio9_4 = (RadioButton) findViewById(R.id.radio9_4);

        textView10=(TextView) findViewById(R.id.textView10);
        radioGroup10 = (RadioGroup) findViewById(R.id.radioGroup10);
        radio10_1 = (RadioButton) findViewById(R.id.radio10_1);
        radio10_2 = (RadioButton) findViewById(R.id.radio10_2);
        radio10_3 = (RadioButton) findViewById(R.id.radio10_3);
        radio10_4 = (RadioButton) findViewById(R.id.radio10_4);

        linearLayout1=(LinearLayout)findViewById(R.id.linearlayout1);
        linearLayout2=(LinearLayout)findViewById(R.id.linearlayout2);
        linearLayout3=(LinearLayout)findViewById(R.id.linearlayout3);
        linearLayout4=(LinearLayout)findViewById(R.id.linearlayout4);
        linearLayout5=(LinearLayout)findViewById(R.id.linearlayout5);
        linearLayout6=(LinearLayout)findViewById(R.id.linearlayout6);
        linearLayout7=(LinearLayout)findViewById(R.id.linearlayout7);
        linearLayout8=(LinearLayout)findViewById(R.id.linearlayout8);
        linearLayout9=(LinearLayout)findViewById(R.id.linearlayout9);
        linearLayout10=(LinearLayout)findViewById(R.id.linearlayout10);



        btn_start.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                sendRequest();
                linearLayout1.setVisibility(View.VISIBLE);
                linearLayout2.setVisibility(View.VISIBLE);
                linearLayout3.setVisibility(View.VISIBLE);
                linearLayout4.setVisibility(View.VISIBLE);
                linearLayout5.setVisibility(View.VISIBLE);
                linearLayout6.setVisibility(View.VISIBLE);
                linearLayout7.setVisibility(View.VISIBLE);
                linearLayout8.setVisibility(View.VISIBLE);
                linearLayout9.setVisibility(View.VISIBLE);
                linearLayout10.setVisibility(View.VISIBLE);
                btn.setVisibility(View.VISIBLE);


            }

        });

        if(requestQueue == null){

            //리퀘스트큐 생성 (MainActivit가 메모리에서 만들어질 때 같이 생성이 될것이다.
            requestQueue = Volley.newRequestQueue(getApplicationContext());
        }



    }


    public void sendRequest(){
        String url = "http://172.30.1.41:5000/return_all_quiz";


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
                        makeJSON(response);
                        print_jsonArray();
                        //println(response);

                    }

                },
                new Response.ErrorListener(){ //에러발생시 호출될 리스너 객체
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        System.out.println("에러 => "+ error.getMessage());
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

        };

        //아래 add코드처럼 넣어줄때 Volley라고하는게 내부에서 캐싱을 해준다, 즉, 한번 보내고 받은 응답결과가 있으면
        //그 다음에 보냈을 떄 이전 게 있으면 그냥 이전거를 보여줄수도  있다.
        //따라서 이렇게 하지말고 매번 받은 결과를 그대로 보여주기 위해 다음과같이 setShouldCache를 false로한다.
        //결과적으로 이전 결과가 있어도 새로 요청한 응답을 보여줌
        request.setShouldCache(false); //이전 결과 있어도 새로 요청하여 응답 보여준다

        requestQueue.add(request);

    }

    public void makeJSON(String res){
        //응답 받은 json 문자열을 jsonarray로 가공하는 함수
        try{
            JSONArray data_list=new JSONArray(res);

            for(int i=0;i<data_list.length();i++){

                JSONObject dt=data_list.getJSONObject(i);
                Data_jsonObject j_dt=new Data_jsonObject(dt.getString("question"),dt.getString("ans_1"),dt.getString("ans_2"),dt.getString("ans_3"),dt.getString("ans_4"),dt.getString("ans"));
                data.add(j_dt);

            }

        }catch(JSONException e){


        }


    }


    public void print_jsonArray(){

        ArrayList<String> question = new ArrayList<String>();
        ArrayList<String> ans_1=new ArrayList<String>();
        ArrayList<String> ans_2=new ArrayList<String>();
        ArrayList<String> ans_3=new ArrayList<String>();
        ArrayList<String> ans_4=new ArrayList<String>();
        ArrayList<String> ans=new ArrayList<String>();


        for (int i = 0; i < 10; i++) {
            Data_jsonObject dt = data.get(i);
            question.add(dt.getQuestion());
            ans_1.add(dt.getAns_1());
            ans_2.add(dt.getAns_2());
            ans_3.add(dt.getAns_3());
            ans_4.add(dt.getAns_4());
            ans.add(dt.getAns());
        }


        textView1.setText(question.get(0));
        radio1_1.setText(ans_1.get(0));
        radio1_2.setText(ans_2.get(0));
        radio1_3.setText(ans_3.get(0));
        radio1_4.setText(ans_4.get(0));

        textView2.setText(question.get(1));
        radio2_1.setText(ans_1.get(1));
        radio2_2.setText(ans_2.get(1));
        radio2_3.setText(ans_3.get(1));
        radio2_4.setText(ans_4.get(1));

        textView3.setText(question.get(2));
        radio3_1.setText(ans_1.get(2));
        radio3_2.setText(ans_2.get(2));
        radio3_3.setText(ans_3.get(2));
        radio3_4.setText(ans_4.get(2));

        textView4.setText(question.get(3));
        radio4_1.setText(ans_1.get(3));
        radio4_2.setText(ans_2.get(3));
        radio4_3.setText(ans_3.get(3));
        radio4_4.setText(ans_4.get(3));

        textView5.setText(question.get(4));
        radio5_1.setText(ans_1.get(4));
        radio5_2.setText(ans_2.get(4));
        radio5_3.setText(ans_3.get(4));
        radio5_4.setText(ans_4.get(4));

        textView6.setText(question.get(5));
        radio6_1.setText(ans_1.get(5));
        radio6_2.setText(ans_2.get(5));
        radio6_3.setText(ans_3.get(5));
        radio6_4.setText(ans_4.get(5));

        textView7.setText(question.get(6));
        radio7_1.setText(ans_1.get(6));
        radio7_2.setText(ans_2.get(6));
        radio7_3.setText(ans_3.get(6));
        radio7_4.setText(ans_4.get(6));

        textView8.setText(question.get(7));
        radio8_1.setText(ans_1.get(7));
        radio8_2.setText(ans_2.get(7));
        radio8_3.setText(ans_3.get(7));
        radio8_4.setText(ans_4.get(7));

        textView9.setText(question.get(8));
        radio9_1.setText(ans_1.get(8));
        radio9_2.setText(ans_2.get(8));
        radio9_3.setText(ans_3.get(8));
        radio9_4.setText(ans_4.get(8));

        textView10.setText(question.get(9));
        radio10_1.setText(ans_1.get(9));
        radio10_2.setText(ans_2.get(9));
        radio10_3.setText(ans_3.get(9));
        radio10_4.setText(ans_4.get(9));

        int check[]= new int[]{0,0,0,0,0,0,0,0,0,0};

        radioGroup1.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio1_1 +1 == Integer.parseInt(ans.get(0))) {
                    tot +=1; check[0]=1;
                }
            }
        });
        radioGroup2.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio2_1 +1 == Integer.parseInt(ans.get(1))) {
                    tot +=1; check[1]=1;
                }
            }
        });
        radioGroup3.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio3_1 +1 == Integer.parseInt(ans.get(2))) {
                    tot +=1; check[2]=1;
                }
            }
        });
        radioGroup4.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio4_1 +1 == Integer.parseInt(ans.get(3))) {
                    tot+=1; check[3]=1;
                }
            }
        });
        radioGroup5.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio5_1 +1 == Integer.parseInt(ans.get(4))) {
                    tot +=1; check[4]=1;
                }
            }
        });
        radioGroup6.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio6_1 +1 == Integer.parseInt(ans.get(5))) {
                    tot +=1; check[5]=1;
                }
            }
        });
        radioGroup7.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio7_1 +1 == Integer.parseInt(ans.get(6))) {
                    tot +=1; check[6]=1;
                }
            }
        });
        radioGroup8.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio8_1 +1 == Integer.parseInt(ans.get(7))) {
                    tot+=1; check[7]=1;
                }
            }
        });
        radioGroup9.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio9_1 +1 == Integer.parseInt(ans.get(8))) {
                    tot +=1; check[8]=1;
                }
            }
        });
        radioGroup10.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            public void onCheckedChanged(RadioGroup group, int checkedId) {
                String result;
                if (checkedId-R.id.radio10_1 +1 == Integer.parseInt(ans.get(9))) {
                    tot+=1; check[9]=1;
                }
            }
        });

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), answer.class);
                Toast.makeText(getApplicationContext(),
                        " 문제 풀기 완료",
                        Toast.LENGTH_SHORT).show();
                intent.putExtra("tot",  Integer.toString(tot));
                startActivity(intent);
            }
        });

    }


}

