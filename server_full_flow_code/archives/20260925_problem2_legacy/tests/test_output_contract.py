import unittest
import numpy as np
from problem2.output_contract import project_intensity


class OutputContractTests(unittest.TestCase):
    def test_evaluation_uses_submission_output_rule(self):
        from unittest.mock import patch
        from tests.test_model_contracts import ModelContracts
        from problem2.validation import evaluate
        case=ModelContracts();case.setUp();case.batch['id']=['a','b']
        with patch('problem2.validation.loader',return_value=[case.batch]):
            frame,_=evaluate(case.model,None,{'seed':1,'device':'cpu','output_rule':'class_constrained'})
        self.assertIn('raw_intensity',frame)
        np.testing.assert_array_equal(np.sign(frame.intensity).astype(int)+1,frame['class'])

    def test_projection_preserves_legal_values_and_repairs_conflicts(self):
        classes=np.array([0,1,2,0,2,1])
        raw=np.array([-2.,.2,1.5,.8,-.6,0.])
        value=project_intensity(classes,raw)
        np.testing.assert_array_equal(np.sign(value).astype(int)+1,classes)
        np.testing.assert_allclose(value,[-2.,0.,1.5,-1e-6,1e-6,0.])
        np.testing.assert_array_equal(project_intensity(classes,value),value)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError): project_intensity([3],[0.])
        with self.assertRaises(ValueError): project_intensity([0],[np.nan])


if __name__ == '__main__': unittest.main()
