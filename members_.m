function[y]=members_(x, f)
%used in the journal
   
    w=zeros(1,5);
    % w(1)=f1(x);
    % w(2)=f2(x);
    % w(3)=f3(x);
    % w(4)=f4(x);
    % w(5)=f5(x);

    w(1)=f1(x(1));
    w(2)=f2(x(2));
    w(3)=f3(x(3));
    w(4)=f4(x(4));
    w(5)=f5(x(5));

    y=sum(w.*f)/sum(w);
end

function[w1]=f1(x)
    if x<-2
        w1=1;
    elseif x>=-2 && x<=-1
        w1=-(x+1);
    else
        w1=0;
    end
end

function[w2]=f2(x)
    if x<-1.5
        w2=0;
    elseif x>=-1.5 && x<=-1.0
        w2=(2)*(x+1.5);
    elseif x>-1.0 && x<=-0.5
        w2=-(2)*(x+0.5);
    else
        w2=0;
    end
end

function[w3]=f3(x)
    if x<-1
        w3=0;
    elseif x>=-1 && x<=0
        w3=(1)*(x+1);
    elseif x>0 && x<=1
        w3=-(1)*(x-1);
    else
        w3=0;
    end
end

function[w4]=f4(x)
    if x<0.5
        w4=0;
    elseif x>=0.5 && x<=1
        w4=(2)*(x-0.5);
    elseif x>1 && x<=1.5
        w4=-(2)*(x-1.5);
    else
        w4=0;
    end
end

function[w5]=f5(x)
    if x<1
        w5=0;
    elseif x>=1 && x<=1.5
        w5=(2)*(x-1);
    else
        w5=1;
    end
end