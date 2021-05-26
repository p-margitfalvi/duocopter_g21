classdef CustomKalman < handle
    properties
        A;
        B;
        C;
        D;
        
        Q;
        R;
        P = [1, 0; 0, 1];
        
        x = [0; 0];
        y_hat = 0;
    end
    methods
        
        function obj = CustomKalman(A, B, C, Q, R)
            obj.A = A;
            obj.B = B;
            obj.C = C;
            %obj.D = D;
            
            obj.Q = Q;
            obj.R = R;
            
            %obj.P = inv(obj.C)*obj.R*inv(obj.C');
            obj.P = [1, 0; 0, 1];
        end
        
        function x_hat = predict(obj, y, u)
            
           % Prediction for state vector and covariance:
           obj.x = obj.A*obj.x + obj.B*u;
           obj.P = obj.A * obj.P * obj.A' + obj.Q;
           
           % Compute Kalman gain factor:
           K = obj.P*obj.C'/(obj.C*obj.P*obj.C'+obj.R);
           
           % Correction based on observation:
           obj.x = obj.x + K*(y-obj.C*obj.x);
           obj.P = obj.P - K*obj.C*obj.P;
            
            %y_hat = obj.C*obj.x_hat + obj.D*u + obj.My*(y - obj.C*obj.x_hat - obj.D*u);
            %obj.x_hat = obj.x_hat + obj.Mx*(y - obj.C*obj.x_hat - obj.D*u);
            %x_hat = obj.A*obj.x_hat + obj.B*u + obj.L*(y - obj.C*obj.x_hat - obj.D*u);
            %obj.x_hat = x_hat;
            x_hat = obj.x;
        end
        
    end
end