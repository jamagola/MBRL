function[state]=findingnemo(x)

    if x<4
        disp("Invalid entry!!")
        state = 0;
    else
        state = 1;
        for i=2:floor(x/2)
            if ((x-i)<100) && ((i)<100)
                A = isPrime_(x-i);
                B = isPrime_(i);
                if A && B
                    state = 0;
                    break;
                else
                    state = 1;
                end
            end
        end
    end
end





function[state] = isPrime_(x)
    state = 1;
    for i=2:floor(x/2)
        if mod(x,i) == 0
            state = 0;
            break;
        else
            state = 1;
        end
    end
end