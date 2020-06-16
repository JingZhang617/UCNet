%%
function Fmeasure = Fmeasure_calu(sMap,gtMap,gtsize, threshold)
%threshold =  2* mean(sMap(:)) ;
%if ( threshold > 1 )
%    threshold = 1;
%end

EPS = 1e-4;
%max(max(sMap))
%threshold

Label3 = zeros( gtsize );
Label3( sMap>=threshold ) = 1;

NumRec = sum(sum(Label3));
%NumRec = length( find( Label3==1 ) );
LabelAnd = Label3 & gtMap;
%NumAnd = length( find ( LabelAnd==1 ) );
NumAnd = sum(sum(LabelAnd));
num_obj = sum(sum(gtMap));

if NumAnd == 0
    PreFtem = 0;
    RecallFtem = 0;
    FmeasureF = 0;
else
    PreFtem = (NumAnd + EPS)/(NumRec + EPS);
    RecallFtem = (NumAnd + EPS)/(num_obj + EPS);
    FmeasureF = ( ( 1.3* PreFtem * RecallFtem) / ( .3 * PreFtem + RecallFtem ) );
end
Fmeasure = [PreFtem, RecallFtem, FmeasureF];

