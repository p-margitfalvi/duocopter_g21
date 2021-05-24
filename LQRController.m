classdef LQRController < handle
    properties
        K;
        waypoints;
        cur_waypoint_idx = 1;
        y_prev = -1;
        t_prev = -1;
        drop_velocity = -2;
        drop_y = 0.1;
        
        drop_thrust = 0.4;
        
        logs;
    end
    methods
        
        function obj = LQRController(K, waypoints)
            obj.K = K;
            obj.waypoints = waypoints;
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
            v = (y - obj.y_prev) / (t - obj.t_prev);
            
            obj.y_prev = y;
            obj.t_prev = t;
            
            thrust = -obj.K*[v; -err];
            
            if obj.waypoints.Data(wpt) < obj.drop_y && y < 0.6 && v < obj.drop_velocity
                % Reduce velocity so as to not hit too hard
                % Max drop velocity should be -3
                %thrust = obj.drop_thrust;
                v_err = v - obj.drop_velocity;
                thrust = -v_err;
            end
            
            thrust = max(min(thrust, 1), 0);
        end
    end
end