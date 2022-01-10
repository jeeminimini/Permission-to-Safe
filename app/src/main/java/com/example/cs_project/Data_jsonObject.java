package com.example.cs_project;

public class Data_jsonObject {

    private String question;

    private String ans_1;
    private String ans_2;
    private String ans_3;
    private String ans_4;
    private String ans;

    public Data_jsonObject(String question,String ans_1,String ans_2,String ans_3,String ans_4,String ans) {



        this.question = question;
        this.ans_1=ans_1;
        this.ans_2=ans_2;
        this.ans_3=ans_3;
        this.ans_4=ans_4;
        this.ans=ans;

    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public void setAns_1(String ans_1) {
        this.ans_1 = ans_1;
    }

    public void setAns_2(String ans_2) {
        this.ans_2 = ans_2;
    }

    public void setAns_3(String ans_3) {
        this.ans_3 = ans_3;
    }
    public void setAns_4(String ans_4) {
        this.ans_4 = ans_4;
    }

    public void setAns(String ans) {
        this.ans = ans;
    }


    public String getQuestion() {
        return question;
    }

    public String getAns_1() {
        return ans_1;
    }

    public String getAns_2() {
        return ans_2;
    }

    public String getAns_3() {
        return ans_3;
    }

    public String getAns_4() {
        return ans_4;
    }

    public String getAns() {
        return ans;
    }

    @Override
    public String toString() {
        return "Data_JsonObject{" +
                "question='" + question + '\'' +
                ", ans_1='" + ans_1 + '\'' +
                ", ans_2='" + ans_2 + '\'' +
                ", ans_3='" + ans_3 + '\'' +
                ", ans_4='" + ans_4 + '\'' +
                ", ans='" + ans + '\'' +
                '}';
    }
}