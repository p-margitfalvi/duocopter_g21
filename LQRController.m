classdef LQRController < handle
    properties
        K;
        waypoints;
        cur_waypoint_idx = 1;
        y_prev = -1;
        y_prev_prev = 0;
        t_prev = -1;
        thrust_prev = 0;
        v_est = 0;
        thrust_est = 0;
        
        drop_velocity = -2;
        drop_y = 0.1;
        
        drop_thrust = 0.4;
        kalman = CustomKalman([0,-3.69279232872370;1*5e-4,0], [38.8714981970916;0], [0,1], [0, 0; 0, 0], 1E-2^2);
        
        logs;
    end
    methods
        
        function obj = LQRController(K, waypoints)
            obj.K = K;
            obj.waypoints = waypoints;
            
            obj.logs.t = [];
            obj.logs.v_est = [];
        end
        
        function cur_waypoint_idx = updateCurWpt(obj, t)
            if obj.waypoints.Time(obj.cur_waypoint_idx) <= t
                cur_waypoint_idx = min(obj.cur_waypoint_idx + 1, numel(obj.waypoints.Time));
                obj.cur_waypoint_idx = cur_waypoint_idx;
            else
                cur_waypoint_idx = obj.cur_waypoint_idx;
            end
        end
        
        function thrust = calculate(obj, y, t)
            wpt = obj.updateCurWpt(t);
            
            err = obj.waypoints.Data(wpt) - y;
            %v = (y - obj.y_prev) / (t - obj.t_prev);
            v = (3*y - 4*obj.y_prev + obj.y_prev_prev) / (2*(t - obj.t_prev));
            
            obj.y_prev_prev = obj.y_prev;
            obj.y_prev = y;
            obj.t_prev = t;
            
            obj.v_est = 0.12*(v - obj.v_est) + obj.v_est;
            v = obj.v_est;
            %x_hat = obj.kalman.predict(y, obj.thrust_prev);
            %v = x_hat(1);
            
            obj.logs.v_est = [obj.logs.v_est, v];
            obj.logs.t = [obj.logs.t, t];
            thrust = -obj.K*[v; -err];
            
            obj.thrust_est = 0.5*(thrust - obj.thrust_est) + obj.thrust_est;
            thrust = obj.thrust_est;
            %{
            if obj.waypoints.Data(wpt) < obj.drop_y && y < 0.6 && v < obj.drop_velocity
                % Reduce velocity so as to not hit too hard
                % Max drop velocity should be -3
                %thrust = obj.drop_thrust;
                v_err = v - obj.drop_velocity;
                thrust = -v_err;
            end
            %}
            
            thrust = max(min(thrust, 1), 0);
            obj.thrust_prev = thrust;
        end
    end
end