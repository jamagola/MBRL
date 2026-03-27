function[y]=members_u(x, f)
%used in the journal
   
    w=zeros(1,5);
    w(1)=f1(x);
    w(2)=f2(x);
    w(3)=f3(x);
    w(4)=f4(x);
    w(5)=f5(x);

    y=sum(w.*f)/sum(w);
end

function[w1]=f1(x)
    if x<-10
        w1=1;
    elseif x>=-10 && x<=-5
        w1=-(1/5)*(x+5);
    else
        w1=0;
    end
end

function[w2]=f2(x)
    if x<-10
        w2=0;
    elseif x>=-10 && x<=-5
        w2=(1/5)*(x+10);
    elseif x>-5 && x<=0
        w2=-(1/5)*(x);
    else
        w2=0;
    end
end

function[w3]=f3(x)
    if x<-5
        w3=0;
    elseif x>=-5 && x<=0
        w3=(1/5)*(x+5);
    elseif x>0 && x<=5
        w3=-(1/5)*(x-5);
    else
        w3=0;
    end
end

function[w4]=f4(x)
    if x<0
        w4=0;
    elseif x>=0 && x<=5
        w4=(1/5)*(x);
    elseif x>5 && x<=10
        w4=-(1/5)*(x-10);
    else
        w4=0;
    end
end

function[w5]=f5(x)
    if x<5
        w5=0;
    elseif x>=5 && x<=10
        w5=(1/5)*(x-5);
    else
        w5=1;
    end
end