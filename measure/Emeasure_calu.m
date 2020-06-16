%%
function EmeasureScore = Emeasure_calu(sMap,gtMap,gtsize, threshold)
%threshold =  2* mean(sMap(:)) ;
%if ( threshold > 1 )
%    threshold = 1;
%end

EPS = 1e-4;
%max(max(sMap))
%threshold

Label3 = zeros( gtsize );
Label3( sMap>=threshold ) = 1;

EmeasureScore = Emeasure(Label3,gtMap);

