
% Note
% Before running this make sure RLmodelBased.m is successfully executed
%vidfile = VideoWriter('testmovie.mp4','MPEG-4');
%open(vidfile);
    for i = 1:10:num_episodes
        im=contourf(xx,yy,QQ(:,:,i));
        zlabel('Q');
        xlabel('action');
        ylabel('states');
        %title(string(i));
        colorbar;
        pause(0.1);
        %writeVideo(vidfile, im);
    end
%close(vidfile)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure() 
contourf(xx,yy,QQ(:,:,1));
 zlabel('Q');
 xlabel('action');
 ylabel('states');
 colorbar;

%pause(0.01);
%figure()
%surf(xx,yy,QQ(:,:,300))