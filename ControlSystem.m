classdef ControlSystem < handle
    properties
        gains;
        waypoints;
        cur_waypoint_idx = 1;
        y_prev = -1;
        t_prev = -1;
        firstIter = true;
        
        err_accum = 0;
        err_prev = 0;
        
        logs;
    end
    methods
        
        function obj = ControlSystem(gains, waypoints)
            obj.gains = gains;
            obj.waypoints = waypoints;
        end
        
        function cur_waypoint_idx = updateCurWpt(obj, t)
            if obj.waypoints.Time(obj.cur_waypoint_idx) <= t
                cur_waypoint_idx = min(obj.cur_waypoint_idx + 1, numel(obj.waypoints.Time));
                obj.cur_waypoint_idx = cur_waypoint_idx;
                obj.reset()
            else
                cur_waypoint_idx = obj.cur_waypoint_idx;
            end
        end
        
        function thrust = calculate(obj, y, t)
            wpt = obj.updateCurWpt(t);
            
            %{
            v_req = (obj.waypoints.Data(wpt) - y) / (obj.waypoints.Time(wpt) - t);
            
            if ~obj.firstIter
                obj.firstIter = false;
                v_cur = (y - obj.y_prev)/(t - obj.t_prev); % TODO: Use more points for smoother derivative
            else
                v_cur = 0;
            end
            %}
            
            err = obj.waypoints.Data(wpt) - y;
            err_deriv = (err - obj.err_prev)/(t - obj.t_prev);
            obj.err_accum = obj.err_accum + err*(t - obj.t_prev);
            
            obj.err_prev = err;
            obj.y_prev = y;
            obj.t_prev = t;
            
            thrust = obj.gains(1) * err + obj.gains(2)*obj.err_accum + obj.gains(3)*err_deriv;
            
            thrust = max(min(thrust, 1), 0);
        end
        
        function reset(obj)
            obj.err_prev = 0;
            obj.err_accum = 0;
        end
    end
end