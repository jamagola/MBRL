function[x,y_1,y_2,u_1,u_2]=workOrderTest(N,D,force,u_1,u_2,y_1,y_2)
% used in the Journal    
    U=[force;u_1;u_2];
    Y=[y_1;y_2];
    x=N(2:end)*U(1:end-1)-D(2:end)*Y;
   
    u_2=u_1;
    u_1=force;
    y_2=y_1;
    y_1=x;

end